'use strict';

define([
        'jquery',
        'lodash',
        'api/RequestError'
    ],
    function ($, _, RequestError) {
        //TODO: Env stub
        var Env = {
            get: function(url) {
                return 'api/' + url
            }
        };

        class Client {
            getConfig(config, cb) {
                // TODO: ES6 style (get rid off slf and function)
                var self = this;

                cb = cb || _.noop();
                self._get(Env.get('config'), {

                }, {
                    onError: (res) => cb(new RequestError(res)),
                    onComplete: (res) => {
                        if (res.status == "ok") {
                            cb(null, res.data);
                        } else {
                            cb(new RequestError(res, true), null);
                        }
                    }
                });
            }

            saveConfig(config, cb) {
                // TODO: ES6 style (get rid off slf and function)
                var self = this;

                cb = cb || _.noop();
                self._post(Env.get('config'), {
                    config: config
                }, {
                    onError: (res) => cb(new RequestError(res)),
                    onComplete: (res) => {
                        if (res.status == "ok") {
                            cb(null, res.data);
                        } else {
                            cb(new RequestError(res, true), null);
                        }
                    }
                });
            }

            train(cb) {
                // TODO: ES6 style (get rid off slf and function)
                var self = this;

                cb = cb || _.noop();
                self._post(Env.get('train'), {

                }, {
                    onError: (res) => cb(new RequestError(res)),
                    onComplete: (res) => {
                        if (res.status == "ok") {
                            cb(null, res.data);
                        } else {
                            cb(new RequestError(res, true), null);
                        }
                    }
                });
            }

            getTrainProgress(cb) {
                // TODO: ES6 style (get rid off slf and function)
                var self = this;

                cb = cb || _.noop();
                self._get(Env.get('train_progress'), {

                }, {
                    onError: (res) => cb(new RequestError(res)),
                    onComplete: (res) => {
                        if (res.status == "ok") {
                            cb(null, res.data);
                        } else {
                            cb(new RequestError(res, true), null);
                        }
                    }
                });
            }

            predict(image, cb) {
                // TODO: ES6 style (get rid off slf and function)
                var self = this;

                cb = cb || _.noop();
                self._post(Env.get('predict'), {
                    image: image
                }, {
                    onError: (res) => cb(new RequestError(res)),
                    onComplete: (res) => {
                        if (res.status == "ok") {
                            cb(null, res.data);
                        } else {
                            cb(new RequestError(res, true), null);
                        }
                    }
                });
            }



            _get(url, data, callbacks) {
                var self = this;
                $.ajax({
                    type: "GET",
                    contentType: "application/json",
                    url: url,
                    data: data
                })
                    .done(function (msg) {
                        self.logData(url, msg);
                        callbacks.onComplete(msg);
                    })
                    .fail(function (error) {
                        callbacks.onError(error);
                    });
            }

            _post(url, data, callbacks) {
                var self = this;
                $.ajax({
                    type: "POST",
                    contentType: "application/json",
                    url: url,
                    dataType: "json",
                    data: JSON.stringify(data)
                })
                    .done(function (msg) {
                        self.logData(url, msg);
                        callbacks.onComplete(msg);
                    })
                    .fail(function (error) {
                        callbacks.onError(error);
                    });
            }

            _postForm(url, data, callbacks) {
                var self = this;
                $.ajax({
                    type: "POST",
                    url: url,
                    dataType: "json",
                    data: data
                })
                    .done(function (msg) {
                        self.logData(url, msg);
                        callbacks.onComplete(msg);
                    })
                    .fail(function (error) {
                        callbacks.onError(error);
                    });
            }

            logData(url, res) {
                console.log("[" + url + "] Message received: ", res);
            }
        }

        return new Client();
    }
);