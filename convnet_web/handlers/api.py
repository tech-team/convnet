from convnet_web.handlers import BaseHandler


class ApiHandler(BaseHandler):

    def api_respond(self, status='ok', data=None):
        d = dict(status=status)
        if data is not None:
            d['data'] = None
        return self.write_json(d)

    def api_ok(self, data):
        return self.api_respond('ok', data)

    def api_error(self, data):
        return self.api_respond('error', data)


class ApiConfig(ApiHandler):
    def post(self):
        req = self.request.body_json

        config = dict(
            learning_rate=self.get_arg('learning_rate', field_type=float),
            momentim=self.get_arg('momentum', field_type=float),
            batch_size=self.get_arg('batch_size', field_type=int),
            iterations_count=self.get_arg('iterations_count', field_type=int),
        )

        layers = self.get_arg('layers')
        pass
