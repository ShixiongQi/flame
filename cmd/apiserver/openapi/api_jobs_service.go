/*
 * Job REST API
 *
 * No description provided (generated by Openapi Generator https://github.com/openapitools/openapi-generator)
 *
 * API version: 1.0.0
 * Generated by: OpenAPI Generator (https://openapi-generator.tech)
 */

package openapi

import (
	"context"
	"errors"
	"net/http"
	"strconv"

	"go.uber.org/zap"

	"wwwin-github.cisco.com/eti/fledge/pkg/objects"
	"wwwin-github.cisco.com/eti/fledge/pkg/util"
)

// JobsApiService is a service that implents the logic for the JobsApiServicer
// This service should implement the business logic for every endpoint for the JobsApi API.
// Include any external packages or services that will be required by this service.
type JobsApiService struct {
}

// NewJobsApiService creates a default api service
func NewJobsApiService() JobsApiServicer {
	return &JobsApiService{}
}

// GetJobs - Get list of all the jobs by the user or based on designId.
func (s *JobsApiService) GetJobs(ctx context.Context, user string, designId string, getType string, limit int32) (ImplResponse, error) {
	//TODO - validate the input
	zap.S().Debugf("Get list of all jobs for user: %s | getType: %s | designId: %s", user, getType, designId)

	//create controller request
	uriMap := map[string]string{
		"user":     user,
		"designId": designId,
		"type":     getType,
		"limit":    strconv.Itoa(int(limit)),
	}
	url := CreateURI(util.GetJobsEndPoint, uriMap)

	//send get request
	responseBody, err := util.HTTPGet(url)

	//response to the user
	if err != nil {
		return Response(http.StatusInternalServerError, nil), errors.New("get jobs list request failed")
	}
	var resp []objects.JobInfo
	err = util.ByteToStruct(responseBody, &resp)
	return Response(http.StatusOK, resp), err
}