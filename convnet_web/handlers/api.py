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

    def write_error(self, status_code, **kwargs):
        self.api_error({
            'code': status_code,
            'reason': self._reason,
        })


class ApiConfig(ApiHandler):
    def post(self):
        req = self.request.body_json

        arg = self.get_arg('test')
        arg = self.get_arg('test1')
        arg = self.get_arg('test1.hello')
        arg = self.get_arg('test1.hello2')
        # config = dict(
        #     learning_rate=self.get_arg('learning_rate', field_type=float),
        #     momentim=self.get_arg('momentum', field_type=float),
        #     batch_size=self.get_arg('batch_size', field_type=int),
        #     iterations_count=self.get_arg('iterations_count', field_type=int),
        # )
        #
        # layers = self.require_arg('layers')

        pass
