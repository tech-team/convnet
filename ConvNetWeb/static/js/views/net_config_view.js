
define([
    'backbone',
    'jquery'
], function(Backbone, $) {
    var NetConfigView = Backbone.View.extend({
        // cache the template
        //tpl: _.template($('#net-config-template').html()),

        events: {
            'keypress .edit': 'update',
        },

        initialize: function() {
            this.$el = $('#net-config');
        },

        render: function() {
            //this.$el.html(this.tpl(this.model.attributes));
            //this.input = this.$('.edit');
            return this;
        },

        update: function(e) {
            // executed on each keypress when in todo edit mode,
            // but we'll wait for enter to get in action
        }
    });

    return NetConfigView;
});