import ray
ray.init()

@ray.remote
class GM:
    def __init__(self):
        self.data = 1

    def get_data(self):
        return self.data

class GM2:
    def __init__(self):
        self.data = 0

    def set(self, data):
        self.data = data

gm = GM.options(name="GM").remote()
gm = ray.get_actor("GM")
gm_local = GM2()
gm_local.set(ray.get(gm.get_data.remote()))
print(gm_local.data)

gm = GM()
print(gm.data)
