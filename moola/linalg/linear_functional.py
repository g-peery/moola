from vector import Vector

class LinearFunctional(Vector):

    def __call__(self, d):
        return self.dot(d)

    def riesz_reprensentation(self):
        return self
