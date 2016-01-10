
define([
    'backbone',
    'jquery',
    'handlebars'
], function(Backbone, $, Handlebars) {
    var NetConfigView = Backbone.View.extend({

        el: '#net-config',
        //template: Handlebars.compile($('#config-template').html()),

        events: {
            'keypress .edit': 'update',
        },

        initialize: function() {

        },

        render: function() {
            this.$el.html(this.template(this.model.attributes));
            this.input = this.$('.edit');
            return this;
        },

        update: function(e) {
            // executed on each keypress when in todo edit mode,
            // but we'll wait for enter to get in action
        }
    });

    return NetConfigView;
});