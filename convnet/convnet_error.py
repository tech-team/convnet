
class ConvNetError(Exception):
    def __init__(self, *args, **kwargs):
        super(ConvNetError, self).__init__(*args, **kwargs)