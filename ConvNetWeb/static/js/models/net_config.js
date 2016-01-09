define([
    'backbone'
], function(Backbone) {
    var NetConfig = Backbone.Model.extend({
        defaults: {
            learnRate: 0
        }
    });

    return NetConfig;
});
