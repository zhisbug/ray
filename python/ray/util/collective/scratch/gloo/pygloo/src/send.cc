#include <collective.h>
// #include <gloo/send.h>
#include <iostream>
namespace pygloo {

template <typename T>
void send(const std::shared_ptr<gloo::Context> &context, intptr_t sendbuf,
          size_t size, int peer) {
  printf("??");

  T *input_ptr = reinterpret_cast<T *>(sendbuf);

  auto inputBuffer = context->createUnboundBuffer(input_ptr, size * sizeof(T));
  printf("??");
  inputBuffer->send(peer, context->rank);
  // inputBuffer->send(peer, 0, 0, size * sizeof(T));
  inputBuffer->waitSend(context->getTimeout());
}

void send_wrapper(const std::shared_ptr<gloo::Context> &context,
                    intptr_t sendbuf, size_t size,
                    glooDataType_t datatype, int peer) {
  switch (datatype) {
  case glooDataType_t::glooInt8:
    send<int8_t>(context, sendbuf, size, peer);
    break;
  case glooDataType_t::glooUint8:
    send<uint8_t>(context, sendbuf, size, peer);
    break;
  case glooDataType_t::glooInt32:
    send<int32_t>(context, sendbuf, size, peer);
    break;
  case glooDataType_t::glooUint32:
    send<uint32_t>(context, sendbuf, size, peer);
    break;
  case glooDataType_t::glooInt64:
    send<int64_t>(context, sendbuf, size, peer);
    break;
  case glooDataType_t::glooUint64:
    send<uint64_t>(context, sendbuf, size, peer);
    break;
  case glooDataType_t::glooFloat16:
    send<gloo::float16>(context, sendbuf, size, peer);
    break;
  case glooDataType_t::glooFloat32:
    send<float_t>(context, sendbuf, size, peer);
    break;
  case glooDataType_t::glooFloat64:
    send<double_t>(context, sendbuf, size, peer);
    break;
  default:
    throw std::runtime_error("Unhandled dataType");
  }
}
} // pygloo