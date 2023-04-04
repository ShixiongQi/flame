// Copyright 2022 Cisco Systems, Inc. and its affiliates
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package app

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	backoff "github.com/cenkalti/backoff/v4"
	"go.uber.org/zap"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/openapi/constants"
	pbNotify "github.com/cisco-open/flame/pkg/proto/notification"
	"github.com/cisco-open/flame/pkg/restapi"
	"github.com/cisco-open/flame/pkg/util"

	// MongoDB deps
	"go.mongodb.org/mongo-driver/bson"
    "go.mongodb.org/mongo-driver/mongo"
    "go.mongodb.org/mongo-driver/mongo/gridfs"
    "go.mongodb.org/mongo-driver/mongo/options"
	"io/ioutil"
	"bytes"
)

const (
	workDir       = "/flame/work"
	pythonBin     = "python3"
	taskPyFile    = "main.py"
	metaFile      = "meta.json"
	logFilePrefix = "task"
	logFileExt    = "log"

	keyTaskId = "taskid"
	keyRole   = "role"
)

type taskHandler struct {
	apiserverEp string
	notifierEp  string
	name        string
	jobId       string
	taskId      string
	taskKey     string

	stream pbNotify.JobEventRoute_GetJobEventClient

	// role of the task given to flamelet
	role string

	state  openapi.JobState
	cancel context.CancelFunc

	grpcDialOpt grpc.DialOption

	mongoClient *mongo.Client
}

func newTaskHandler(apiserverEp string, notifierEp string, name string, taskId string, taskKey string,
	bInsecure bool, bPlain bool) *taskHandler {
	var grpcDialOpt grpc.DialOption

	if bPlain {
		grpcDialOpt = grpc.WithTransportCredentials(insecure.NewCredentials())
	} else {
		tlsCfg := &tls.Config{}
		if bInsecure {
			zap.S().Warn("Warning: allow insecure connection\n")

			tlsCfg.InsecureSkipVerify = true
			http.DefaultTransport.(*http.Transport).TLSClientConfig = tlsCfg
		}
		grpcDialOpt = grpc.WithTransportCredentials(credentials.NewTLS(tlsCfg))
	}

	conn := InitiateMongoClient() // https://www.mongodb.com/blog/post/quick-start-golang--mongodb--a-quick-look-at-gridfs

	return &taskHandler{
		apiserverEp: apiserverEp,
		notifierEp:  notifierEp,
		name:        name,
		taskId:      taskId,
		taskKey:     taskKey,
		state:       openapi.READY,
		grpcDialOpt: grpcDialOpt,
		mongoClient: conn,
	}
}

func InitiateMongoClient() *mongo.Client {
	if os.Getenv("ROLE") != "aggregator" {
		return &mongo.Client{}
	}

	zap.S().Infof("%v is connecting to MongoDB...", os.Getenv("ROLE"))

    var err error
    var client *mongo.Client

	uri := "mongodb://flame-mongodb-headless:27017/?replicaSet=rs0"
    opts := options.Client()
    opts.ApplyURI(uri)
    opts.SetMaxPoolSize(5)

    if client, err = mongo.Connect(context.Background(), opts); err != nil {
        zap.S().Warnf(err.Error())
    }
    return client
}

// start connects to the notifier via grpc and handles notifications from the notifier
func (t *taskHandler) start() {
	go t.doStart()
}

func (t *taskHandler) doStart() {
	pauseTime := 10 * time.Second

	for {
		expBackoff := backoff.NewExponentialBackOff()
		expBackoff.MaxElapsedTime = 5 * time.Minute // max wait time: 5 minutes
		err := backoff.Retry(t.connect, expBackoff)
		if err != nil {
			zap.S().Fatalf("Cannot connect with notifier: %v", err)
		}

		t.do()

		// if connection is broken right after connection is made, this can cause
		// too many connection/disconnection events. To migitage that, add some static
		// pause time.
		time.Sleep(pauseTime)
	}
}

func (t *taskHandler) connect() error {
	// dial server
	conn, err := grpc.Dial(t.notifierEp, t.grpcDialOpt)
	if err != nil {
		zap.S().Debugf("Cannot connect with notifier: %v", err)
		return err
	}

	client := pbNotify.NewJobEventRouteClient(conn)
	in := &pbNotify.JobTaskInfo{
		Id:       t.taskId,
		Hostname: t.name,
	}

	// setup notification stream
	stream, err := client.GetJobEvent(context.Background(), in)
	if err != nil {
		zap.S().Debugf("Open stream error: %v", err)
		return err
	}

	t.stream = stream
	zap.S().Infof("Connected with notifier at %s", t.notifierEp)

	return nil
}

func (t *taskHandler) do() {
	for {
		resp, err := t.stream.Recv()
		if err != nil {
			zap.S().Errorf("Failed to receive notification: %v", err)
			break
		}

		t.dealWith(resp)
	}

	zap.S().Info("Disconnected from notifier")
}

// newNotification acts as a handler and calls respective functions based on the response type to act on the received notifications.
func (t *taskHandler) dealWith(in *pbNotify.JobEvent) {
	switch in.GetType() {
	case pbNotify.JobEventType_START_JOB:
		t.startJob(in.JobId)

	case pbNotify.JobEventType_STOP_JOB:
		t.stopJob(in.JobId)

	case pbNotify.JobEventType_UPDATE_JOB:
		t.updateJob(in.JobId)

	case pbNotify.JobEventType_UNKNOWN_EVENT_TYPE:
		fallthrough
	default:
		zap.S().Errorf("Invalid message type: %s", in.GetType())
	}
}

// startJob starts the application on the agent
func (t *taskHandler) startJob(jobId string) {
	zap.S().Infof("Received start job request on job %s", jobId)

	if t.state == openapi.RUNNING {
		zap.S().Infof("Task is already running job %s", jobId)
		return
	}
	// assign job ID to task handler
	t.jobId = jobId

	if os.Getenv("ROLE") == "aggregator" {
		zap.S().Infof("Aggregator loads taskFile and configFile from MongoDB")

		if os.MkdirAll(workDir, util.FilePerm0755) != nil {
			zap.S().Warnf("Failed to create path: %v", workDir)
		}

		if os.MkdirAll(filepath.Join(workDir, t.role), util.FilePerm0755) != nil {
			zap.S().Warnf("Failed to create path: %v", filepath.Join(workDir, t.role))
		}

		taskFilePath := filepath.Join(workDir, t.role, taskPyFile)
		configFilePath := filepath.Join(workDir, util.TaskConfigFile)
		if t.DownloadFile(taskFilePath, "taskFile") == 0 || t.DownloadFile(configFilePath, "configFile") == 0 {
			zap.S().Infof("MongoDB doesn't have taskFile and/or configFile... Getting files from API server")

			filePaths, err := t.getTask()
			if err != nil {
				zap.S().Warnf("Failed to download payload: %v", err)
				return
			}
		
			err = t.prepareTask(filePaths)
			if err != nil {
				zap.S().Warnf("Failed to prepare task")
				return
			}
		}

		metaFilePath := filepath.Join(workDir, metaFile)
		if t.DownloadFile(metaFilePath, "metaFile") == 1 {
			zap.S().Infof("Download meta.json to (%s)", metaFilePath)
		} else {
			zap.S().Infof("No meta.json on MongoDB. This could be 1st round")
		}

		go t.runTask()

	} else {

		filePaths, err := t.getTask()
		if err != nil {
			zap.S().Warnf("Failed to download payload: %v", err)
			return
		}
	
		err = t.prepareTask(filePaths)
		if err != nil {
			zap.S().Warnf("Failed to prepare task")
			return
		}
	
		go t.runTask()

	}

	// TODO: implement updateTaskStatus method
}

func (t *taskHandler) getTask() ([]string, error) {
	// construct URL
	uriMap := map[string]string{
		constants.ParamJobID:  t.jobId,
		constants.ParamTaskID: t.taskId,
		constants.ParamKey:    t.taskKey,
	}
	url := restapi.CreateURL(t.apiserverEp, restapi.GetTaskEndpoint, uriMap)

	code, taskMap, err := restapi.HTTPGetMultipart(url)
	if err != nil || restapi.CheckStatusCode(code) != nil {
		errMsg := fmt.Sprintf("Failed to fetch task - code: %d, error: %v", code, err)
		zap.S().Warnf(errMsg)
		return nil, fmt.Errorf(errMsg)
	}

	filePaths := make([]string, 0)
	for fileName, data := range taskMap {
		filePath := filepath.Join("/tmp", fileName)
		err = os.WriteFile(filePath, data, util.FilePerm0755)
		if err != nil {
			zap.S().Warnf("Failed to save %s: %v\n", fileName, err)
			return nil, err
		}

		filePaths = append(filePaths, filePath)

		zap.S().Infof("Downloaded %s successfully", fileName)
	}

	return filePaths, nil
}

func (t *taskHandler) prepareTask(filePaths []string) error {
	err := os.MkdirAll(workDir, util.FilePerm0755)
	if err != nil {
		return err
	}

	var fileDataList []util.FileData
	var file *os.File
	configFilePath := ""
	configFound := false
	codeFound := false
	for _, filePath := range filePaths {
		if strings.Contains(filePath, util.TaskConfigFile) {
			configFound = true

			configFilePath = filePath
		} else if strings.Contains(filePath, util.TaskCodeFile) {
			codeFound = true

			file, err = os.Open(filePath)
			if err != nil {
				return fmt.Errorf("failed to open %s: %v", filePath, err)
			}

			fileDataList, err = util.UnzipFile(file)
			if err != nil {
				return err
			}
		}
	}

	if !configFound || !codeFound {
		return fmt.Errorf("either %s or %s not found", util.TaskConfigFile, util.TaskCodeFile)
	}

	err = t.prepareConfig(configFilePath)
	if err != nil {
		return err
	}

	err = t.prepareCode(fileDataList)
	if err != nil {
		return err
	}

	return nil
}

func (t *taskHandler) prepareConfig(configFilePath string) error {
	// copy config file to work directory
	input, err := os.ReadFile(configFilePath)
	if err != nil {
		return fmt.Errorf("failed to open config file %s: %v", configFilePath, err)
	}

	tmp := make(map[string]interface{})
	err = json.Unmarshal(input, &tmp)
	if err != nil {
		return fmt.Errorf("failed to unmarhsal config data")
	}
	t.role = tmp[keyRole].(string)

	// add task id in the config
	tmp[keyTaskId] = t.taskId

	// marshall the updated config
	input, err = json.Marshal(&tmp)
	if err != nil {
		return fmt.Errorf("failed to marhsal config data")
	}

	dstFilePath := filepath.Join(workDir, util.TaskConfigFile)
	err = os.WriteFile(dstFilePath, input, util.FilePerm0644)
	if err != nil {
		return fmt.Errorf("failed to copy config file: %v", err)
	}

	return nil
}

func (t *taskHandler) prepareCode(fileDataList []util.FileData) error {
	// copy code files to work directory
	for _, fileData := range fileDataList {
		dirPath := filepath.Join(workDir, filepath.Dir(fileData.FullName))
		err := os.MkdirAll(dirPath, util.FilePerm0755)
		if err != nil {
			return fmt.Errorf("failed to create directory: %v", err)
		}

		filePath := filepath.Join(dirPath, fileData.BaseName)
		err = os.WriteFile(filePath, []byte(fileData.Data), util.FilePerm0644)
		if err != nil {
			return fmt.Errorf("failed to unzip file %s: %v", filePath, err)
		}
	}

	return nil
}

func (t *taskHandler) stopJob(jobId string) {
	if t.jobId != jobId {
		zap.S().Warnf("stop request on a wrong job %s", jobId)
		return
	}

	if t.state != openapi.RUNNING {
		zap.S().Warnf("job  %s is not in a running state", jobId)
		return
	}

	if t.cancel == nil {
		zap.S().Warnf("cancel function not specified for job %s", jobId)
		return
	}

	t.cancel()
}

func (t *taskHandler) updateJob(jobId string) (string, error) {
	zap.S().Infof("not yet implemented; received update job request on job %s", jobId)
	return "", nil
}

func (t *taskHandler) runTask() {
	taskFilePath := filepath.Join(workDir, t.role, taskPyFile)
	configFilePath := filepath.Join(workDir, util.TaskConfigFile)

	ctx, cacnel := context.WithCancel(context.Background())
	defer cacnel()
	t.cancel = cacnel

	// TODO: run the task in different user group with less privilege
	cmd := exec.CommandContext(ctx, pythonBin, taskFilePath, configFilePath)
	zap.S().Debugf("Running task with command: %v", cmd)

	file, err := os.Create(t.getLogfilePath())
	if err != nil {
		zap.S().Errorf("Failed to create a log file: %v", err)
		return
	}
	defer file.Close()

	cmd.Stdout = file
	cmd.Stderr = file

	zap.S().Infof("Starting task for job %s", t.jobId)
	// set running state in advance
	// this is for keeping state transition simple, meaning that flamelet always
	// set running state first before making transition to one of three states
	// -- failed, terminated, completed
	t.updateTaskStatus(openapi.RUNNING, "")

	getLog := func() string {
		bytesToRead := 10000000000
		log := ""
		log, err = t.readLastNBytesFromFile(t.getLogfilePath(), bytesToRead)
		if err != nil {
			log = fmt.Sprintf("failed to access log: %v", err)
		}
		return log
	}

	err = cmd.Start()
	if err != nil {
		zap.S().Errorf("Failed to start task: %v", err)
		t.updateTaskStatus(openapi.FAILED, getLog())
		return
	}

	err = cmd.Wait()
	if err != nil {
		if ctx.Err() == context.Canceled {
			// cancel() function was called
			zap.S().Infof("Task execution canceled for job %s: %v", t.jobId, err)
			t.updateTaskStatus(openapi.TERMINATED, getLog())
		} else {
			zap.S().Infof("Task execution failed for job %s: %v", t.jobId, err)
			t.updateTaskStatus(openapi.FAILED, getLog())
		}
	} else {
		zap.S().Infof("Task execution successful for job %s", t.jobId)
		t.updateTaskStatus(openapi.COMPLETED, getLog())
	}

	if os.Getenv("ROLE") == "aggregator" {
		zap.S().Infof("Aggregator saves taskFile and configFile to MongoDB")
		t.UploadFile(taskFilePath, "taskFile")
		t.UploadFile(configFilePath, "configFile")

		metaFilePath := filepath.Join(workDir, metaFile)
		zap.S().Infof("Upload meta.json (%s)", metaFilePath)
		t.UploadFile(metaFilePath, "metaFile")

		// TODO: clean up mongoDB entry when training is done
	}
}

func (t *taskHandler) getLogfilePath() string {
	logFileName := fmt.Sprintf("%s-%s.%s", logFilePrefix, t.jobId, logFileExt)
	logFilePath := filepath.Join(util.LogDirPath, logFileName)

	return logFilePath
}

func (t *taskHandler) readLastNBytesFromFile(filePath string, nbytes int) (string, error) {
	fileHandle, err := os.Open(filePath)
	if err != nil {
		return "", fmt.Errorf("fail to open log file")
	}
	defer fileHandle.Close()

	var cursor int64
	stat, _ := fileHandle.Stat()
	filesize := stat.Size()
	if filesize > int64(nbytes) {
		cursor = -int64(nbytes)
	} else {
		cursor = -filesize
	}

	_, err = fileHandle.Seek(cursor, io.SeekEnd)
	if err != nil {
		return "", fmt.Errorf("fail to seek log file")
	}

	buf := make([]byte, 0-cursor)
	_, err = fileHandle.Read(buf)
	if err != nil {
		return "", fmt.Errorf("fail to read log file")
	}

	return string(buf), nil
}

func (t *taskHandler) updateTaskStatus(state openapi.JobState, log string) {
	// update state in the task handler
	t.state = state

	// construct URL
	uriMap := map[string]string{
		constants.ParamJobID:  t.jobId,
		constants.ParamTaskID: t.taskId,
	}
	url := restapi.CreateURL(t.apiserverEp, restapi.UpdateTaskStatusEndPoint, uriMap)

	taskStatus := openapi.TaskStatus{
		State: state,
		Log:   log,
	}

	code, _, err := restapi.HTTPPut(url, taskStatus, "application/json")
	if err != nil || restapi.CheckStatusCode(code) != nil {
		zap.S().Warnf("Failed to update a task status - code: %d, error: %v\n", code, err)
		return
	}
}

func (t *taskHandler) UploadFile(file, filename string) {
    conn := t.mongoClient
    bucket, err := gridfs.NewBucket(
        conn.Database("dbName"),
    )
    if err != nil {
        zap.S().Fatal(err)
        os.Exit(1)
    }

	uploadStream, err := bucket.OpenUploadStream(
        filename,
    )
    if err != nil {
        zap.S().Infoln(err)
        os.Exit(1)
    }
    defer uploadStream.Close()

	data, err := ioutil.ReadFile(file)
    if err != nil {
        zap.S().Fatal(err)
    }

    fileSize, err := uploadStream.Write(data)
    if err != nil {
        zap.S().Fatal(err)
        os.Exit(1)
    }
    zap.S().Infof("Write file to DB was successful. File size: %d Bytes\n", fileSize)
}

func (t *taskHandler) DownloadFile(filePath, fileName string) int {
    conn := t.mongoClient

    // For CRUD operations, here is an example
    db := conn.Database("dbName")
    fsFiles := db.Collection("fs.files")
    ctx, _ := context.WithTimeout(context.Background(), 10 * time.Second)

    var results bson.M // bson.M{{"filename", "taskFile"}}
    err := fsFiles.FindOne(ctx, bson.M{}).Decode(&results)
    if err != nil {
		if err == mongo.ErrNoDocuments {
			// This error means your query did not match any documents.
			return 0
		}
        zap.S().Fatal(err)
    }
    
    zap.S().Infoln(results) // print out the results

    bucket, _ := gridfs.NewBucket(
        db,
    )

    var buf bytes.Buffer
    dStream, err := bucket.DownloadToStreamByName(fileName, &buf)
    if err != nil {
        zap.S().Fatal(err)
    }
    zap.S().Infof("File size to download: %v\n", dStream)

    ioutil.WriteFile(filePath, buf.Bytes(), 0644)

	return 1
}
