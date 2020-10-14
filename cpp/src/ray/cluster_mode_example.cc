
#include <ray/api.h>
#include <ray/api/ray_config.h>
#include <ray/experimental/default_worker.h>

using namespace ray::api;

/// general function of user code
int Return1() { return 1; }
int Plus1(int x) { return x + 1; }
int Plus(int x, int y) { return x + y; }

/// a class of user code
class Counter {
 public:
  int count;

  Counter(int init) { count = init; }

  Counter *FactoryCreate() { return new Counter(0); }
  static Counter *FactoryCreate(int init) { return new Counter(init); }
  static Counter *FactoryCreate(int init1, int init2) { return new Counter(init1+init2); }
  /// non static function
  int Add(int x) {
    count += x;
    return count;
  }
};

void example(){
  ray::api::RayConfig::GetInstance()->run_mode = RunMode::CLUSTER;
  ray::api::RayConfig::GetInstance()->lib_name = "";
  Ray::Init();

  /// put and get object
  auto obj = Ray::Put(12345);
  auto get_result = *(Ray::Get(obj));
  EXPECT_EQ(12345, get_result);

  auto task_obj = Ray::Task(Plus1, 5).Remote();
  int task_result = *(Ray::Get(task_obj));
  EXPECT_EQ(6, task_result);

  ActorHandle<Counter> actor0 = Ray::Actor(Counter::FactoryCreate, 0,1).Remote();
  auto actor_object = actor0.Task(&Counter::Add, 5).Remote();
  int actor_task_result = *(Ray::Get(actor_object));
  EXPECT_EQ(6, actor_task_result);
  
  /// general function remote call（args passed by value）
  auto r0 = Ray::Task(Return1).Remote();
  auto r1 = Ray::Task(Plus1, 1).Remote();
  auto r2 = Ray::Task(Plus, 1, 2).Remote();

  int result0 = *(Ray::Get(r0));
  int result1 = *(Ray::Get(r1));
  int result2 = *(Ray::Get(r2));

  std::cout << "Ray::call with value results: " << result0 << " " << result1 << " "
            << result2 << std::endl;

  /// general function remote call（args passed by reference）
  auto r3 = Ray::Task(Return1).Remote();
  auto r4 = Ray::Task(Plus1, 3).Remote();
  auto r5 = Ray::Task(Plus, 4, 1).Remote();

  int result3 = *(Ray::Get(r3));
  int result4 = *(Ray::Get(r4));
  int result5 = *(Ray::Get(r5));
  std::cout << "Ray::call with reference results: " << result3 << " " << result4 << " "
            << result5 << std::endl;

  /// create actor and actor function remote call
  ActorHandle<Counter> actor = Ray::Actor(Counter::FactoryCreate, 0).Remote();
  auto r6 = actor.Task(&Counter::Add, 5).Remote();
  auto r7 = actor.Task(&Counter::Add, 1).Remote();
  auto r8 = actor.Task(&Counter::Add, 1).Remote();
  auto r9 = actor.Task(&Counter::Add, 8).Remote();

  int result6 = *(Ray::Get(r6));
  int result7 = *(Ray::Get(r7));
  int result8 = *(Ray::Get(r8));
  int result9 = *(Ray::Get(r9));

  std::cout << "Ray::call with actor results: " << result6 << " " << result7 << " "
            << result8 << " " << result9 << std::endl;
  Ray::Shutdown();
}

int main(int argc, char **argv) {
//////// DO NOT REMOVE THESE LINES. ////////
  const char *default_worker_magic = "is_default_worker";
  if (argc > 1 &&
      memcmp(argv[argc - 1], default_worker_magic, strlen(default_worker_magic)) == 0) {
    default_worker_main(argc, argv);
    return 0;
  }
//////// DO NOT REMOVE THESE LINES. ////////
  example();
  return 0;
}
