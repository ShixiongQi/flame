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

/*
 * Flame REST API
 *
 * No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)
 *
 * API version: 1.0.0
 * Generated by: OpenAPI Generator (https://openapi-generator.tech)
 */

package apiserver

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"go.uber.org/zap"

	"github.com/cisco-open/flame/pkg/openapi"
	"github.com/cisco-open/flame/pkg/restapi"
)

// DesignCodesApiService is a service that implents the logic for the DesignCodesApiServicer
// This service should implement the business logic for every endpoint for the DesignCodesApi API.
// Include any external packages or services that will be required by this service.
type DesignCodesApiService struct {
}

// NewDesignCodesApiService creates a default api service
func NewDesignCodesApiService() openapi.DesignCodesApiServicer {
	return &DesignCodesApiService{}
}

// CreateDesignCode - Upload a new design code
func (s *DesignCodesApiService) CreateDesignCode(ctx context.Context, user string, designId string,
	fileName string, fileVer string, fileData *os.File) (openapi.ImplResponse, error) {
	zap.S().Debugf("Received CreateDesignCode POST request: %s | %s | %s | %s", user, designId, fileName, fileVer)

	// Don't forget to close the temp file
	defer fileData.Close()

	uriMap := map[string]string{
		"user":     user,
		"designId": designId,
	}
	url := restapi.CreateURL(HostEndpoint, restapi.CreateDesignCodeEndPoint, uriMap)

	// "fileName", "fileVer" and "fileData" are names of variables used in openapi specification
	kv := map[string]io.Reader{
		"fileName": strings.NewReader(fileName),
		"fileVer":  strings.NewReader(fileVer),
		"fileData": fileData,
	}

	// create multipart/form-data
	buf, writer, err := restapi.CreateMultipartFormData(kv)
	if err != nil {
		respErr := fmt.Errorf("create multipart/form-data failed: %v", err)
		return openapi.Response(http.StatusInternalServerError, nil), respErr
	}

	// send post request
	resp, err := http.Post(url, writer.FormDataContentType(), buf)
	if err != nil {
		var msg string
		body, _ := io.ReadAll(resp.Body)
		_ = json.Unmarshal(body, &msg)
		respErr := fmt.Errorf("create new code request failed: %s", msg)
		return openapi.Response(http.StatusInternalServerError, nil), respErr
	}
	defer resp.Body.Close()

	if err = restapi.CheckStatusCode(resp.StatusCode); err != nil {
		var msg string
		body, _ := io.ReadAll(resp.Body)
		_ = json.Unmarshal(body, &msg)
		return openapi.Response(resp.StatusCode, nil), fmt.Errorf("%s", msg)
	}

	return openapi.Response(http.StatusCreated, nil), nil
}

// DeleteDesignCode - Delete a zipped design code file owned by user
func (s *DesignCodesApiService) DeleteDesignCode(ctx context.Context, user string, designId string,
	version string) (openapi.ImplResponse, error) {
	//create controller request
	uriMap := map[string]string{
		"user":     user,
		"designId": designId,
		"version":  version,
	}
	url := restapi.CreateURL(HostEndpoint, restapi.DeleteDesignCodeEndPoint, uriMap)

	// send Delete request
	code, body, err := restapi.HTTPDelete(url, "", "application/json")
	errResp, retErr := errorResponse(code, body, err)
	if retErr != nil {
		return errResp, retErr
	}

	return openapi.Response(http.StatusOK, body), nil
}

// GetDesignCode - Get a zipped design code file owned by user
func (s *DesignCodesApiService) GetDesignCode(ctx context.Context, user string, designId string,
	version string) (openapi.ImplResponse, error) {
	zap.S().Debugf("Get design code for user: %s | designId: %s | version: %s", user, designId, version)

	//create controller request
	uriMap := map[string]string{
		"user":     user,
		"designId": designId,
		"version":  version,
	}
	url := restapi.CreateURL(HostEndpoint, restapi.GetDesignCodeEndPoint, uriMap)

	// send get request
	code, body, err := restapi.HTTPGet(url)
	errResp, retErr := errorResponse(code, body, err)
	if retErr != nil {
		return errResp, retErr
	}

	return openapi.Response(http.StatusOK, body), nil
}

// UpdateDesignCode - Update a design code
func (s *DesignCodesApiService) UpdateDesignCode(ctx context.Context, user string, designId string, version string,
	fileName string, fileVer string, fileData *os.File) (openapi.ImplResponse, error) {
	// TODO - update UpdateDesignCode with the required logic for this service method.
	// Add api_design_codes_service.go to the .openapi-generator-ignore to avoid overwriting this service
	// implementation when updating open api generation.

	//TODO: Uncomment the next line to return response Response(200, {}) or use other options such as http.Ok ...
	//return Response(200, nil),nil

	//TODO: Uncomment the next line to return response Response(0, Error{}) or use other options such as http.Ok ...
	//return Response(0, Error{}), nil

	return openapi.Response(http.StatusNotImplemented, nil), errors.New("UpdateDesignCode method not implemented")
}
