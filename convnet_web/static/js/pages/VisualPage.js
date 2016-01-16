'use strict';

define([
    'jquery',
    'jquery-ui',
    'util/Templater'
], function($, jui, Templater) {
    class VisualPage {
        constructor() {
            this.$el = $('#visual-page');
            this.$canvas = this.$el.find('#display');
            this.ctx = this.$canvas[0].getContext('2d');
        }
        
        draw(layers, desiredImgW, desiredImgH) {
            let imgData = this.ctx.createImageData(this.ctx.canvas.width, this.ctx.canvas.height);

            for (let l = 0; l < layers.length; ++l) {
                let layer = layers[l];
                for (let z = 0; z < layer.length; ++z) {
                    let image = layer[z];

                    const upSampleY = image.length / desiredImgW;
                    for (let y = 0; y < desiredImgH; ++y) {
                        let line = image[Math.floor(y * upSampleY)];

                        const upSampleX = line.length / desiredImgH;
                        for (let x = 0; x < desiredImgH; ++x) {
                            var xOffset = z*50;
                            var yOffset = l*50;

                            var r = ((y + yOffset) * imgData.width + x + xOffset) * 4;
                            var g = r + 1;
                            var b = g + 1;
                            var a = b + 1;

                            var color = line[Math.floor(x * upSampleX)];

                            imgData.data[r] = color;
                            imgData.data[g] = color;
                            imgData.data[b] = color;
                            imgData.data[a] = 255;
                        }
                    }
                }
            }

            this.ctx.putImageData(imgData, 0, 0);
        }
    }

    return VisualPage;
});