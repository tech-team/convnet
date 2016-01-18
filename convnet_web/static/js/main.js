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
                    "layers": [
                        {
                            "type": "conv",
                            "filter_size": 5,
                            "filters_count": 8,
                            "stride": 1,
                            "zero_padding": 0
                        },
                        {
                            "type": "relu",
                            "activation": "max"
                        },
                        {
                            "type": "pool",
                            "filter_size": 2,
                            "stride": 1
                        },
                        {
                            "type": "fc",
                            "neurons_count": 10
                        }
                    ]
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
