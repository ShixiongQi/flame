// Copyright (c) 2021 Cisco Systems, Inc. and its affiliates
// All rights reserved
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
 * Fledge REST API
 *
 * No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)
 *
 * API version: 1.0.0
 * Generated by: OpenAPI Generator (https://openapi-generator.tech)
 */

package controller

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"time"

	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/cmd/controller/app/database"
	"wwwin-github.cisco.com/eti/fledge/cmd/controller/app/job"
	"wwwin-github.cisco.com/eti/fledge/pkg/openapi"
)

const (
	defaultWaitTime = 10 * time.Second
)

// JobsApiService is a service that implents the logic for the JobsApiServicer
// This service should implement the business logic for every endpoint for the JobsApi API.
// Include any external packages or services that will be required by this service.
type JobsApiService struct {
	jobEventQ *job.EventQ
}

// NewJobsApiService creates a default api service
func NewJobsApiService(jobEventQ *job.EventQ) openapi.JobsApiServicer {
	return &JobsApiService{jobEventQ: jobEventQ}
}

// CreateJob - Create a new job specification
func (s *JobsApiService) CreateJob(ctx context.Context, user string, jobSpec openapi.JobSpec) (openapi.ImplResponse, error) {
	jobStatus, err := database.CreateJob(user, jobSpec)
	if err != nil {
		errMsg := fmt.Sprintf("failed to create a new job: %v", err)
		zap.S().Debug(errMsg)
		return openapi.Response(http.StatusInternalServerError, nil), fmt.Errorf(errMsg)
	}

	return openapi.Response(http.StatusCreated, jobStatus), nil
}

// DeleteJob - Delete job specification
func (s *JobsApiService) DeleteJob(ctx context.Context, user string, jobId string) (openapi.ImplResponse, error) {
	// TODO - update DeleteJob with the required logic for this service method.
	// Add api_jobs_service.go to the .openapi-generator-ignore to avoid overwriting this service
	// implementation when updating open api generation.

	//TODO: Uncomment the next line to return response Response(200, {}) or use other options such as http.Ok ...
	//return Response(200, nil),nil

	//TODO: Uncomment the next line to return response Response(404, {}) or use other options such as http.Ok ...
	//return Response(404, nil),nil

	//TODO: Uncomment the next line to return response Response(401, {}) or use other options such as http.Ok ...
	//return Response(401, nil),nil

	//TODO: Uncomment the next line to return response Response(0, Error{}) or use other options such as http.Ok ...
	//return Response(0, Error{}), nil

	return openapi.Response(http.StatusNotImplemented, nil), errors.New("DeleteJob method not implemented")
}

// GetJob - Get a job specification
func (s *JobsApiService) GetJob(ctx context.Context, user string, jobId string) (openapi.ImplResponse, error) {
	// TODO - update GetJob with the required logic for this service method.
	// Add api_jobs_service.go to the .openapi-generator-ignore to avoid overwriting this service
	// implementation when updating open api generation.

	//TODO: Uncomment the next line to return response Response(200, JobSpec{}) or use other options such as http.Ok ...
	//return Response(200, JobSpec{}), nil

	//TODO: Uncomment the next line to return response Response(0, Error{}) or use other options such as http.Ok ...
	//return Response(0, Error{}), nil

	return openapi.Response(http.StatusNotImplemented, nil), errors.New("GetJob method not implemented")
}

// GetJobStatus - Get job status of a given jobId
func (s *JobsApiService) GetJobStatus(ctx context.Context, user string, jobId string) (openapi.ImplResponse, error) {
	// TODO - update GetJobStatus with the required logic for this service method.
	// Add api_jobs_service.go to the .openapi-generator-ignore to avoid overwriting this service
	// implementation when updating open api generation.

	//TODO: Uncomment the next line to return response Response(200, JobStatus{}) or use other options such as http.Ok ...
	//return Response(200, JobStatus{}), nil

	//TODO: Uncomment the next line to return response Response(0, Error{}) or use other options such as http.Ok ...
	//return Response(0, Error{}), nil

	return openapi.Response(http.StatusNotImplemented, nil), errors.New("GetJobStatus method not implemented")
}

// GetJobsStatus - Get status info on all the jobs owned by user
func (s *JobsApiService) GetJobsStatus(ctx context.Context, user string, limit int32) (openapi.ImplResponse, error) {
	// TODO - update GetJobsStatus with the required logic for this service method.
	// Add api_jobs_service.go to the .openapi-generator-ignore to avoid overwriting this service
	// implementation when updating open api generation.

	//TODO: Uncomment the next line to return response Response(200, []JobStatus{}) or use other options such as http.Ok ...
	//return Response(200, []JobStatus{}), nil

	//TODO: Uncomment the next line to return response Response(0, Error{}) or use other options such as http.Ok ...
	//return Response(0, Error{}), nil

	return openapi.Response(http.StatusNotImplemented, nil), errors.New("GetJobsStatus method not implemented")
}

// GetTask - Get a job task for a given job and agent
func (s *JobsApiService) GetTask(ctx context.Context, jobId string, agentId string) (openapi.ImplResponse, error) {
	// TODO - update GetTask with the required logic for this service method.
	// Add api_jobs_service.go to the .openapi-generator-ignore to avoid overwriting this service
	// implementation when updating open api generation.

	//TODO: Uncomment the next line to return response Response(200, Task{}) or use other options such as http.Ok ...
	//return Response(200, Task{}), nil

	//TODO: Uncomment the next line to return response Response(0, Error{}) or use other options such as http.Ok ...
	//return Response(0, Error{}), nil

	return openapi.Response(http.StatusNotImplemented, nil), errors.New("GetTask method not implemented")
}

// UpdateJob - Update a job specification
func (s *JobsApiService) UpdateJob(ctx context.Context, user string, jobId string,
	jobSpec openapi.JobSpec) (openapi.ImplResponse, error) {
	// TODO - update UpdateJob with the required logic for this service method.
	// Add api_jobs_service.go to the .openapi-generator-ignore to avoid overwriting this service
	// implementation when updating open api generation.

	//TODO: Uncomment the next line to return response Response(200, {}) or use other options such as http.Ok ...
	//return Response(200, nil),nil

	//TODO: Uncomment the next line to return response Response(401, {}) or use other options such as http.Ok ...
	//return Response(401, nil),nil

	//TODO: Uncomment the next line to return response Response(0, Error{}) or use other options such as http.Ok ...
	//return Response(0, Error{}), nil

	return openapi.Response(http.StatusNotImplemented, nil), errors.New("UpdateJob method not implemented")
}

// UpdateJobStatus - Update the status of a job
func (s *JobsApiService) UpdateJobStatus(ctx context.Context, user string, jobId string,
	jobStatus openapi.JobStatus) (openapi.ImplResponse, error) {
	// override jobId in the jobStatus
	jobStatus.Id = jobId

	event := job.NewJobEvent(user, jobStatus)
	s.jobEventQ.Enqueue(event)

	select {
	case <-time.After(defaultWaitTime):
		return openapi.Response(http.StatusInternalServerError, nil), fmt.Errorf("response timed out")

	case err := <-event.ErrCh:
		if err != nil {
			errMsg := fmt.Sprintf("failed to update job status to %s: %v", jobStatus.State, err)
			zap.S().Debug(errMsg)
			return openapi.Response(http.StatusInternalServerError, nil), fmt.Errorf(errMsg)
		}

		return openapi.Response(http.StatusOK, nil), nil
	}
}
