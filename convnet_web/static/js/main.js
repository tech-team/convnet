'use strict';

define([
    'jquery',
    'jquery-ui',
    'handlebars',
    'pages/ConfigPage'
], function($, jui, Handlebars,  ConfigPage) {
    // TODO: add Router here

    Handlebars.registerHelper('select', function( value, options ){
        var $el = $('<select />').html( options.fn(this) );
        $el.find('[value="' + value + '"]').attr({'selected':'selected'});
        return $el.html();
    });

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
});