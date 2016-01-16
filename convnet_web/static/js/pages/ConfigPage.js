'use strict';

define([
    'jquery',
    'jquery-ui',
    'util/Templater'
], function($, jui, Templater) {
    class ConfigPage {
        constructor() {
            this.$el = $('.config-page');

            this.templates = {
                common: Templater.compile('#common-props-template'),
                layer: Templater.compile('#layer-template')
            };

            this.targets = {
                common: this.$el.find('#common-props'),
                layers: this.$el.find('#layers-container')
            };

            var $layers = this.$el.find('#layers-container');
            $layers.sortable();

            $layers.on('click', '.remove', (e) => {
                $(e.target).closest('.layer').remove();
            });

            this.$el.find('.add').click((e) => {
                var defaultLayer = {
                    name: 'Layer',
                    type: 'input',
                    w: 5,
                    h: 5,
                    d: 3
                };
                this.addLayer(defaultLayer);
            });

            this.$el.find('.save').click((e) => {
                var config = this.serialize();
                console.log(config);

                // TODO: send to server
                alert('Not implemented');
            });
        }

        render(config) {
            var commonHtml = this.templates.common({
                c: config.common
            });
            this.targets.common.html(commonHtml);

            this.targets.layers.empty();
            for (let layer of config.layers) {
                this.addLayer(layer);
            }
        }

        serialize() {
            var config = {
                common: {},
                layers: []
            };

            this.targets.common.find('.value').each((i, el) => {
                var $el = $(el);
                var key = $el.data('key');
                config.common[key] = $el.val();
            });

            this.targets.layers.find('.layer').each((i, layer) => {
                var $layer = $(layer);

                var data = {};
                $layer.find('.value').each((j, el) => {
                    var $el = $(el);
                    var key = $el.data('key');
                    data[key] = $el.val();
                });
                config.layers.push(data);
            });

            return config;
        }

        addLayer(data) {
            var layerHtml = this.templates.layer({
                l: data
            });

            var $layerHtml = $(layerHtml);
            $layerHtml.appendTo(this.targets.layers);
        }
    }

    return ConfigPage;
});