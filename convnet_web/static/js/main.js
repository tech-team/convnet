'use strict';

define([
    'jquery',
    'jquery-ui',
    'handlebars',
    'pages/ConfigPage',
    'pages/VisualPage'
], function($, jui, Handlebars,  ConfigPage, VisualPage) {
    registerHelpers();

    switch(window.location.pathname) {
        case '/':
        case '/index':
        case '/main':
            visual();
            break;
        case '/config':
            config();
            break;
        default:
            alert('We are lost in space: ' + window.location.pathname);
    }

    function visual() {
        var page = new VisualPage();

        // generate random images
        var layers = [];
        for (let l = 0; l < 10; ++l) {
            let layer = [];
            for (let z = 0; z < 10; ++z) {
                var image = [];
                for (let y = 0; y < 2; ++y) {
                    var line = [];
                    for (let x = 0; x < 2; ++x) {
                        line.push(Math.floor(Math.random() * 255));
                    }
                    image.push(line);
                }
                layer.push(image);
            }
            layers.push(layer);
        }

        page.display.display(layers, 48, 48);
    }

    function config() {
        $(() => {
            var page = new ConfigPage();
            var config = {
                common: {
                    learnRate: 0.01
                },
                layers: [{
                    name: 'Layer',
                    type: 'input',
                    w: 5,
                    h: 5,
                    d: 3
                }, {
                    name: 'Layer',
                    type: 'conv',
                    w: 3,
                    h: 3,
                    d: 2
                }, {
                    name: 'Layer',
                    type: 'relu',
                    w: 3,
                    h: 3,
                    d: 2
                }, {
                    name: 'Layer',
                    type: 'pool',
                    w: 2,
                    h: 2,
                    d: 2
                }, {
                    name: 'Layer',
                    type: 'full',
                    w: 1,
                    h: 1,
                    d: 3
                }]
            };

            page.render(config);
        });
    }

    function registerHelpers() {
        Handlebars.registerHelper('select', function( value, options ){
            var $el = $('<select />').html( options.fn(this) );
            $el.find('[value="' + value + '"]').attr({'selected':'selected'});
            return $el.html();
        });
    }
});
