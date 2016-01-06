require.config({
    urlArgs: "_=",
    baseUrl: "js",
    paths: {
        jquery: "../../bower_components/jquery/dist/jquery",
        lodash: "../../bower_components/lodash/lodash"
    },
    shim: {
        lodash: {
            exports: "_"
        },
        jquery: {
            exports: "$"
        }
    },
    packages: [

    ]
});

define([
    'main'
], function() {
    window.DEBUG = true;
});
