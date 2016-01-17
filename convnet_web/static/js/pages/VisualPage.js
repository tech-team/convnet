'use strict';

define([
    'jquery',
    'jquery-ui',
    'util/Templater',
    'api/Client'
], function($, jui, Templater, Client) {
    class VisualPage {
        constructor() {
            this.$el = $('#visual-page');

            this.display = new Display(this.$el.find('#display'));
            this.imageInput = new ImageInput(this.$el.find('#image-input'));

            this.$progress = this.$el.find('.progress');
            this.$configure = this.$el.find('.configure');
            this.$train = this.$el.find('.train');
            this.$recognize = this.$el.find('.recognize');
            this.$clear = this.$el.find('.clear');
            this.$result = this.$el.find('.result');

            this.$configure.click(() => this.configure());
            this.$train.click(() => this.train());
            this.$recognize.click(() => this.recognize());
            this.$clear.click(() => this.imageInput.clear());
        }

        train() {
            Client.train((error, data) => {
                if (error)
                    return this.onError(error);

                this.pollProgress();
            });
        }

        pollProgress() {
            Client.getTrainProgress((error, data) => {
                if (error)
                    return this.onError(error);

                this.$progress.val(data.progress);
                console.log('Progress: ' + data.progress);

                setTimeout(() => this.pollProgress(), 1000);
            });
        }

        recognize() {
            Client.train((error, data) => {
                if (error)
                    return this.onError(error);

                this.$result.text(data.value);
            });
        }

        configure() {
            // navigate
            window.location.href = '/config';
        }

        onError(error) {
            console.log(error);
            alert(JSON.stringify(error));
        }
    }

    class Display {
        constructor($canvas) {
            this.$canvas = $canvas;
            this.ctx = this.$canvas[0].getContext('2d');
        }

        display(layers, desiredImgW, desiredImgH) {
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

    class ImageInput {
        constructor($canvas) {
            this.$canvas = $canvas;
            this.ctx = this.$canvas[0].getContext('2d');

            this.clear();

            $canvas.mousedown((e) => {
                this.paint = true;
                this.addClick(e.offsetX, e.offsetY);
                this.redraw();
            });

            $canvas.mousemove((e) => {
                if (this.paint) {
                    this.addClick(e.offsetX, e.offsetY, true);
                    this.redraw();
                }
            });

            $canvas.mouseup((e) => this.paint = false);
            $canvas.mouseleave((e) => this.paint = false);
        }

        redraw() {
            var context = this.ctx;

            context.clearRect(0, 0, context.canvas.width, context.canvas.height); // Clears the canvas

            context.strokeStyle = "black";
            context.lineJoin = "round";
            context.lineWidth = 5;

            for(var i = 0; i < this.clickX.length; ++i) {
                context.beginPath();
                if(this.clickDrag[i] && i) {
                    context.moveTo(this.clickX[i - 1], this.clickY[i - 1]);
                } else {
                    context.moveTo(this.clickX[i] - 1, this.clickY[i]);
                }
                context.lineTo(this.clickX[i], this.clickY[i]);
                context.closePath();
                context.stroke();
            }
        }

        clear() {
            this.clickX = [];
            this.clickY = [];
            this.clickDrag = [];
            this.paint = false;

            this.redraw();
        }

        getDataURL() {
            return this.ctx.canvas.toDataURL('image/png', 1);
        }

        addClick(x, y, dragging) {
            this.clickX.push(x);
            this.clickY.push(y);
            this.clickDrag.push(dragging);
        }
    }

    return VisualPage;
});