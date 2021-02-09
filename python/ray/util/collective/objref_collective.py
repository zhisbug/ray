import ray
#Brian's Local Tests. 

#TODO: fix the ray.get thing
#RESOLVED: fixed with invocation wrapper see @test_wrap and method.__ray_invocation_decorator__

#TODO: experiment with other collectives to see if it works.
#Need compute resources to do. 

#TODO: expose the API for other users. 
#TODO: objref.get now promotes to ray store... ask if this functionality works?

class CollectiveActorClass:

    def __init__(self):
        self._pos = {}
        # temp hash for keys into pos.
        self._hash = 1

        def _DO_NOT_DISPATCH_TO_RAY(invocation):
            def _ignore_invocation(args, kwargs):
                pos_objref = args[0]
                return self._pos[pos_objref]  
            return _ignore_invocation

    
    def _gen_pos_ref(self, value):
        #TODO: generate some hash that actually makes sense. 
        objref = str(self._hash)
        self._hash += 1
        return objref

    def set_pos(self, objref, value):
        self._pos[objref] = value

    def promote_to_raystore(self, objref):
        if objref not in self._pos:
            raise KeyError("objref does not exist in {}'s pos store.".format(self))

        # this will automatically do it. 
        return self._pos[objref]


def pos(method):
    """
    POS Wrapper for CollectiveActor methods.
    """

    def _ignore_ray_objref(invocation):
        """
        This decorator is applied on invocation of the _modified function. 
        See __ray_invocation_decorator__ API for more details.
        Essentially, just gets rid of the Ray objref issue we had before. 
        """

        def _ignore_me(args, kwargs):
            return ray.get(invocation(args, kwargs))

        return _ignore_me


    def _modified(*args, **kwargs):
        """
        Stores rsult of method inside CollectiveActor POS instead of ray store. 
        Bypasses serialization => lower latencies for resolving objects. 
        """

        if not len(args) > 0:
            # Brian's notes:
            # We can implement an @pos for non-methods, e.g. standalone functions.
            # let me know if that seems like something we would be interested in
            # doing. So far the API specifications only list for methods. 
            raise TypeError("Must be a method!")

        #first must be object for method
        collective_actor = args[0]

        if not isinstance(collective_actor, CollectiveActorClass):
            raise TypeError("Can only wrap @pos for Collective Actors.")

        rv = method(*args, **kwargs)
        objref = collective_actor._gen_pos_ref(rv)
        collective_actor.set_pos(objref, rv)

        print("DEBUG: stored {} at objref {}".format(
                                                    rv, 
                                                    objref))
        return objref

    # I defined a __ray_pos__ flag for pos methods. 
    _modified.__ray_pos__ = True
    _modified.__ray_invocation_decorator__ = _ignore_ray_objref
    return _modified


def CollectiveActor(cls):

    #Let me know if there's a better way to do this than multiple inheritance. 
    class _CollectiveActor(CollectiveActorClass, cls):
        def __init__(self, *args, **kwargs):
            CollectiveActorClass.__init__(self)
            cls.__init__(self, *args, **kwargs)

    return _CollectiveActor


"""
Testing Code. 
"""
# import cupy as cp
# import ray.util.collective as collective

# @ray.remote(num_gpus=1)
# @CollectiveActor
# class Worker:
#     def __init__(self):
#         self.send = cp.ones((4, ), dtype=cp.float32)
#         self.recv = cp.zeros((4, ), dtype=cp.float32)

#     def setup(self, world_size, rank):
#         collective.init_collective_group("nccl", world_size, rank, "default")
#         return True

#     @pos
#     def compute(self):
#         collective.allreduce(self.send, "default")
#         print(self.send)
#         return self.send

#     def destroy(self):
#         collective.destroy_group()


# if __name__ == "__main__":

#     send = cp.ones((4, ), dtype=cp.float32)

#     ray.init(num_gpus=2)

#     num_workers = 2
#     workers = []
#     init_rets = []
#     for i in range(num_workers):
#         w = Worker.remote()
#         workers.append(w)
#         init_rets.append(w.setup.remote(num_workers, i))
#     _ = ray.get(init_rets)
#     for w in workers:
#         w.compute.remote()

#     objrefs = ray.get([w.compute.remote() for w in workers])
#     print(results)
    # ray.shutdown()
@ray.remote
@CollectiveActor
class MyActor:
    def __init__(self):
        self.buffer = 1

    # get_buffer was a user function that returns self.buffer.
    @pos
    def get_buffer(self):
        buffer = self.buffer
        return buffer


if __name__ == '__main__':
    ray.init(num_gpus=2)
    actor = MyActor.remote()
    pos_ref = actor.get_buffer.remote()
    print(pos_ref)

    ray.shutdown()
    
