require.config({
    urlArgs: "_=",
    baseUrl: "static/js",
    paths: {
        jquery: "../components/jquery/dist/jquery",
        'jquery-ui': '../components/jquery-ui/jquery-ui',
        bootstrap: "../components/bootstrap/dist/js/bootstrap",
        backbone: "../components/backbone/backbone",
        underscore: "../components/underscore/underscore",
        lodash: "../components/lodash/lodash",
        handlebars: "../components/handlebars/handlebars"
    },
    shim: {
        underscore: {
            exports: "_"
        },
        lodash: {
            exports: "_"
        },
        bootstrap: {
            deps: [
                "jquery"
            ]
        },
        backbone: {
            exports: "Backbone",
            deps: [
                "jquery",
                "underscore"
            ]
        }
    },
    packages: [

    ]
});

define([
    'backbone',
    './main'
], function(Backbone, main) {
    Backbone.history.start();
    window.DEBUG = true;
});

