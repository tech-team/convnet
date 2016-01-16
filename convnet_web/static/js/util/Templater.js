define(['jquery', 'handlebars'], function($, Handlebars) {
    return {
        compile: function(selector) {
            var $template = $(selector);
            var str = $template.html();

            var preparedStr = str.replace(/\[\[/g, "\{\{").replace(/]]/g, "\}\}");

            return Handlebars.compile(preparedStr);
        }
    };
});