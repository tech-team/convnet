define([
    'jquery',
    './models/net_config',
    './views/net_config_view'
], function($, NetConfig, NetConfigView) {
    var netConfig = new NetConfig();
    var netConfigView = new NetConfigView({
        model: netConfig
    });
    netConfigView.render();
});