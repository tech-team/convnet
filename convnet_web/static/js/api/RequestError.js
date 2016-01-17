'use strict';
define(['jquery'], function($) {
    class RequestError {
        /**
         * Request Error Object
         * @param resp
         * @param resp.responseJSON
         * @param resp.responseText
         * @param resp.status
         * @param apiError
         */
        constructor(resp, apiError) {
            this.errorType = 0;
            this.data = null;
            this.httpStatus = null;

            this.Type = {
                None: 'none',
                ApiError: 'apiError',
                ServerError: 'serverError'
            };

            if (!apiError) {
                this.httpStatus = resp.status;
                if (resp.responseJSON != null) {
                    this.errorType = this.Type.ApiError;
                    this.data = resp.responseJSON;
                } else {
                    this.errorType = this.Type.ServerError;
                    this.data = resp.responseText;
                }
            } else {
                this.errorType = this.Type.ApiError;
                this.data = resp;
            }

            console.error(this);
        }

        isServerError() {
            return this.errorType == this.Type.ServerError;
        }

        isApiError() {
            return this.errorType == this.Type.ApiError;
        }

        getData() {
            return this.data;
        }

        getHttpStatus() {
            return this.httpStatus;
        }
    }

    return RequestError;
});

