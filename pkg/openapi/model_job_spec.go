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

package openapi

import "github.com/cisco-open/flame/pkg/openapi/constants"

// JobSpec - Job specification
type JobSpec struct {
	Id string `json:"id,omitempty"`

	UserId string `json:"userId,omitempty"`

	DesignId string `json:"designId"`

	SchemaVersion string `json:"schemaVersion"`

	CodeVersion string `json:"codeVersion"`

	DataSpec DataSpec `json:"dataSpec,omitempty"`

	Optimizer Optimizer `json:"optimizer,omitempty"`

	Selector Selector `json:"selector,omitempty"`

	Priority JobPriority `json:"priority,omitempty"`

	Backend CommBackend `json:"backend,omitempty"`

	MaxRunTime int32 `json:"maxRunTime,omitempty"`

	BaseModel BaseModel `json:"baseModel,omitempty"`

	Hyperparameters map[string]interface{} `json:"hyperparameters,omitempty"`

	Dependencies []string `json:"dependencies,omitempty"`
}

// AssertJobSpecRequired checks if the required fields are not zero-ed
func AssertJobSpecRequired(obj JobSpec) error {
	elements := map[string]interface{}{
		constants.ParamDesignID: obj.DesignId,
		"schemaVersion":         obj.SchemaVersion,
		"codeVersion":           obj.CodeVersion,
	}
	for name, el := range elements {
		if isZero := IsZeroValue(el); isZero {
			return &RequiredError{Field: name}
		}
	}

	if err := AssertDataSpecRequired(obj.DataSpec); err != nil {
		return err
	}
	if err := AssertOptimizerRequired(obj.Optimizer); err != nil {
		return err
	}
	if err := AssertSelectorRequired(obj.Selector); err != nil {
		return err
	}
	if err := AssertBaseModelRequired(obj.BaseModel); err != nil {
		return err
	}
	return nil
}

// AssertRecurseJobSpecRequired recursively checks if required fields are not zero-ed in a nested slice.
// Accepts only nested slice of JobSpec (e.g. [][]JobSpec), otherwise ErrTypeAssertionError is thrown.
func AssertRecurseJobSpecRequired(objSlice interface{}) error {
	return AssertRecurseInterfaceRequired(objSlice, func(obj interface{}) error {
		aJobSpec, ok := obj.(JobSpec)
		if !ok {
			return ErrTypeAssertionError
		}
		return AssertJobSpecRequired(aJobSpec)
	})
}
