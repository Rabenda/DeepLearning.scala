package com.thoughtworks.compute

import scala.collection.JavaConverters._
import java.nio.{ByteBuffer, IntBuffer}
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.{AtomicInteger, AtomicReference}

import org.lwjgl.opencl._
import CL10._
import CL12._
import CL11._
import CL20._
import KHRICD._
import org.lwjgl.{BufferUtils, PointerBuffer}
import org.lwjgl.system.MemoryUtil._
import org.lwjgl.system.MemoryStack._
import org.lwjgl.system.Pointer._

import scala.collection.mutable
import com.thoughtworks.compute.Memory.Box
import com.thoughtworks.compute.OpenCL.checkErrorCode
import org.lwjgl.system.jni.JNINativeInterface
import org.lwjgl.system._

import scala.util.control.Exception.Catcher
import scala.util.control.{NonFatal, TailCalls}
import scala.util.control.TailCalls.TailRec
import scala.util.{Failure, Success, Try}
import scalaz.{-\/, Memo, \/, \/-}
import scalaz.syntax.all._
import com.thoughtworks.continuation._
import com.thoughtworks.feature.Factory
import com.thoughtworks.feature.Factory.inject
import com.thoughtworks.feature.mixins.ImplicitsSingleton
import com.thoughtworks.future._
import com.thoughtworks.raii.AsynchronousPool
import com.thoughtworks.raii.asynchronous._
import com.thoughtworks.raii.covariant._
import com.thoughtworks.tryt.covariant._
import shapeless.Witness
import simulacrum.typeclass

import scala.language.higherKinds

/**
  * @author 杨博 (Yang Bo)
  */
object OpenCL {

  var numberOfBuffers = new AtomicInteger(0)

  /** Returns a [[String]] for the C string `address`.
    *
    * @note We don't know the exact charset of the C string. Use [[memASCII]] because lwjgl treats them as ASCII.
    */
  private def decodeString(address: Long): String = memASCII(address)

  /** Returns a [[String]] for the C string `byteBuffer`.
    *
    * @note We don't know the exact charset of the C string. Use [[memASCII]] because lwjgl treats them as ASCII.
    */
  private def decodeString(byteBuffer: ByteBuffer): String = memASCII(byteBuffer)

  @volatile
  var defaultLogger: (String, ByteBuffer) => Unit = { (errorInfo: String, data: ByteBuffer) =>
    // TODO: Add a test for in the case that Context is closed
    Console.err.println(raw"""An OpenCL notify comes out after its corresponding handler is freed
  message: $errorInfo
  data: $data""")
  }
  private val contextCallback: CLContextCallback = CLContextCallback.create(new CLContextCallbackI {
    override def invoke(errInfo: Long, privateInfo: Long, size: Long, userData: Long): Unit = {
      val errorInfo = decodeString(errInfo)
      val data = memByteBuffer(privateInfo, size.toInt)
      memGlobalRefToObject[OpenCL](userData) match {
        case null =>
          defaultLogger(decodeString(errInfo), memByteBuffer(privateInfo, size.toInt))
        case opencl =>
          if (size.isValidInt) {
            opencl.handleOpenCLNotification(decodeString(errInfo), memByteBuffer(privateInfo, size.toInt))
          } else {
            throw new IllegalArgumentException(s"numberOfBytes($size) is too large")
          }
      }
    }
  })
  object Exceptions {
    final class MisalignedSubBufferOffset extends IllegalArgumentException

    final class ExecStatusErrorForEventsInWaitList extends IllegalArgumentException

    final class InvalidProperty extends IllegalArgumentException

    final class PlatformNotFoundKhr extends IllegalStateException

    final class DeviceNotFound extends IllegalArgumentException

    final class DeviceNotAvailable extends IllegalStateException

    final class CompilerNotAvailable extends IllegalStateException

    final class MemObjectAllocationFailure extends IllegalStateException

    final class OutOfResources extends IllegalStateException

    final class OutOfHostMemory extends IllegalStateException

    final class ProfilingInfoNotAvailable extends IllegalStateException

    final class MemCopyOverlap extends IllegalStateException

    final class ImageFormatMismatch extends IllegalStateException

    final class ImageFormatNotSupported extends IllegalStateException

    final class BuildProgramFailure(buildLogs: Map[Long /* device id */, String] = Map.empty)
        extends IllegalStateException({
          buildLogs.view
            .map {
              case (deviceId, buildLog) =>
                s"device $deviceId log:\n$buildLog"
            }
            .mkString("\n")
        })

    final class MapFailure extends IllegalStateException

    final class InvalidValue extends IllegalArgumentException

    final class InvalidDeviceType extends IllegalArgumentException

    final class InvalidPlatform extends IllegalArgumentException

    final class InvalidDevice extends IllegalArgumentException

    final class InvalidContext extends IllegalArgumentException

    final class InvalidQueueProperties extends IllegalArgumentException

    final class InvalidCommandQueue extends IllegalArgumentException

    final class InvalidHostPtr extends IllegalArgumentException

    final class InvalidMemObject extends IllegalArgumentException

    final class InvalidImageFormatDescriptor extends IllegalArgumentException

    final class InvalidImageSize extends IllegalArgumentException

    final class InvalidSampler extends IllegalArgumentException

    final class InvalidBinary extends IllegalArgumentException

    final class InvalidBuildOptions extends IllegalArgumentException

    final class InvalidProgram extends IllegalArgumentException

    final class InvalidProgramExecutable extends IllegalArgumentException

    final class InvalidKernelName extends IllegalArgumentException

    final class InvalidKernelDefinition extends IllegalArgumentException

    final class InvalidKernel extends IllegalArgumentException

    final class InvalidArgIndex extends IllegalArgumentException

    final class InvalidArgValue extends IllegalArgumentException

    final class InvalidArgSize extends IllegalArgumentException

    final class InvalidKernelArgs extends IllegalArgumentException

    final class InvalidWorkDimension extends IllegalArgumentException

    final class InvalidWorkGroupSize extends IllegalArgumentException

    final class InvalidWorkItemSize extends IllegalArgumentException

    final class InvalidGlobalOffset extends IllegalArgumentException

    final class InvalidEventWaitList extends IllegalArgumentException

    final class InvalidEvent extends IllegalArgumentException

    final class InvalidOperation extends IllegalArgumentException

    final class InvalidBufferSize extends IllegalArgumentException

    final class InvalidGlobalWorkSize extends IllegalArgumentException

    final class UnknownErrorCode(errorCode: Int) extends IllegalStateException(s"Unknown error code: $errorCode")

    def fromErrorCode(errorCode: Int): Exception = errorCode match {
      case CL_PLATFORM_NOT_FOUND_KHR                    => new Exceptions.PlatformNotFoundKhr
      case CL_DEVICE_NOT_FOUND                          => new Exceptions.DeviceNotFound
      case CL_DEVICE_NOT_AVAILABLE                      => new Exceptions.DeviceNotAvailable
      case CL_COMPILER_NOT_AVAILABLE                    => new Exceptions.CompilerNotAvailable
      case CL_MEM_OBJECT_ALLOCATION_FAILURE             => new Exceptions.MemObjectAllocationFailure
      case CL_OUT_OF_RESOURCES                          => new Exceptions.OutOfResources
      case CL_OUT_OF_HOST_MEMORY                        => new Exceptions.OutOfHostMemory
      case CL_PROFILING_INFO_NOT_AVAILABLE              => new Exceptions.ProfilingInfoNotAvailable
      case CL_MEM_COPY_OVERLAP                          => new Exceptions.MemCopyOverlap
      case CL_IMAGE_FORMAT_MISMATCH                     => new Exceptions.ImageFormatMismatch
      case CL_IMAGE_FORMAT_NOT_SUPPORTED                => new Exceptions.ImageFormatNotSupported
      case CL_BUILD_PROGRAM_FAILURE                     => new Exceptions.BuildProgramFailure
      case CL_MAP_FAILURE                               => new Exceptions.MapFailure
      case CL_INVALID_VALUE                             => new Exceptions.InvalidValue
      case CL_INVALID_DEVICE_TYPE                       => new Exceptions.InvalidDeviceType
      case CL_INVALID_PLATFORM                          => new Exceptions.InvalidPlatform
      case CL_INVALID_DEVICE                            => new Exceptions.InvalidDevice
      case CL_INVALID_CONTEXT                           => new Exceptions.InvalidContext
      case CL_INVALID_QUEUE_PROPERTIES                  => new Exceptions.InvalidQueueProperties
      case CL_INVALID_COMMAND_QUEUE                     => new Exceptions.InvalidCommandQueue
      case CL_INVALID_HOST_PTR                          => new Exceptions.InvalidHostPtr
      case CL_INVALID_MEM_OBJECT                        => new Exceptions.InvalidMemObject
      case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR           => new Exceptions.InvalidImageFormatDescriptor
      case CL_INVALID_IMAGE_SIZE                        => new Exceptions.InvalidImageSize
      case CL_INVALID_SAMPLER                           => new Exceptions.InvalidSampler
      case CL_INVALID_BINARY                            => new Exceptions.InvalidBinary
      case CL_INVALID_BUILD_OPTIONS                     => new Exceptions.InvalidBuildOptions
      case CL_INVALID_PROGRAM                           => new Exceptions.InvalidProgram
      case CL_INVALID_PROGRAM_EXECUTABLE                => new Exceptions.InvalidProgramExecutable
      case CL_INVALID_KERNEL_NAME                       => new Exceptions.InvalidKernelName
      case CL_INVALID_KERNEL_DEFINITION                 => new Exceptions.InvalidKernelDefinition
      case CL_INVALID_KERNEL                            => new Exceptions.InvalidKernel
      case CL_INVALID_ARG_INDEX                         => new Exceptions.InvalidArgIndex
      case CL_INVALID_ARG_VALUE                         => new Exceptions.InvalidArgValue
      case CL_INVALID_ARG_SIZE                          => new Exceptions.InvalidArgSize
      case CL_INVALID_KERNEL_ARGS                       => new Exceptions.InvalidKernelArgs
      case CL_INVALID_WORK_DIMENSION                    => new Exceptions.InvalidWorkDimension
      case CL_INVALID_WORK_GROUP_SIZE                   => new Exceptions.InvalidWorkGroupSize
      case CL_INVALID_WORK_ITEM_SIZE                    => new Exceptions.InvalidWorkItemSize
      case CL_INVALID_GLOBAL_OFFSET                     => new Exceptions.InvalidGlobalOffset
      case CL_INVALID_EVENT_WAIT_LIST                   => new Exceptions.InvalidEventWaitList
      case CL_INVALID_EVENT                             => new Exceptions.InvalidEvent
      case CL_INVALID_OPERATION                         => new Exceptions.InvalidOperation
      case CL_INVALID_BUFFER_SIZE                       => new Exceptions.InvalidBufferSize
      case CL_INVALID_GLOBAL_WORK_SIZE                  => new Exceptions.InvalidGlobalWorkSize
      case CL_MISALIGNED_SUB_BUFFER_OFFSET              => new Exceptions.MisalignedSubBufferOffset
      case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST => new Exceptions.ExecStatusErrorForEventsInWaitList
      case CL_INVALID_PROPERTY                          => new Exceptions.InvalidProperty

      case _ => new Exceptions.UnknownErrorCode(errorCode)

    }

  }

  def checkErrorCode(errorCode: Int): Unit = {
    errorCode match {
      case CL_SUCCESS =>
      case _          => throw Exceptions.fromErrorCode(errorCode)
    }
  }

  trait UseFirstPlatform {
    @transient
    protected lazy val platformId: Long = {
      val stack = stackPush()
      try {
        val platformIdBuffer = stack.mallocPointer(1)

        println("call clGetPlatformIDs"); checkErrorCode(clGetPlatformIDs(platformIdBuffer, null: IntBuffer))
        platformIdBuffer.get(0)
      } finally {
        stack.close()
      }
    }
  }

  private def deviceIdsByType(platformId: Long, deviceType: Int): Seq[Long] = {
    val Array(numberOfDevices) = {
      val a = Array(0)

      println("call clGetDeviceIDs"); checkErrorCode(clGetDeviceIDs(platformId, deviceType, null, a))
      a
    }
    val stack = stackPush()
    try {
      val deviceIds = stack.mallocPointer(numberOfDevices)

      println("call clGetDeviceIDs"); checkErrorCode(clGetDeviceIDs(platformId, deviceType, deviceIds, null: IntBuffer))
      for (i <- 0 until deviceIds.capacity()) yield {
        val deviceId = deviceIds.get(i)
        deviceId
      }
    } finally {
      stack.close()
    }
  }

  trait UseAllDevices {

    protected val platformId: Long

    @transient
    protected lazy val deviceIds: Seq[Long] = {
      deviceIdsByType(platformId, CL_DEVICE_TYPE_ALL)
    }

  }

  trait UseFirstDevice {

    protected val platformId: Long

    @transient
    protected lazy val deviceIds: Seq[Long] = {
      val allDeviceIds = deviceIdsByType(platformId, CL_DEVICE_TYPE_ALL)
      Seq(allDeviceIds.head)
    }

  }

  trait UseAllGpuDevice {

    protected val platformId: Long

    @transient
    protected lazy val deviceIds: Seq[Long] = {
      deviceIdsByType(platformId, CL_DEVICE_TYPE_GPU)
    }
  }

  trait UseFirstGpuDevice {

    protected val platformId: Long

    @transient
    protected lazy val deviceIds: Seq[Long] = {
      val allDeviceIds = deviceIdsByType(platformId, CL_DEVICE_TYPE_GPU)
      Seq(allDeviceIds.head)
    }
  }
  trait UseFirstCpuDevice {

    protected val platformId: Long

    @transient
    protected lazy val deviceIds: Seq[Long] = {
      val allDeviceIds = deviceIdsByType(platformId, CL_DEVICE_TYPE_CPU)
      Seq(allDeviceIds.head)
    }
  }

  trait UseAllCpuDevice {

    protected val platformId: Long

    @transient
    protected lazy val deviceIds: Seq[Long] = {
      deviceIdsByType(platformId, CL_DEVICE_TYPE_CPU)
    }
  }

  object CommandQueuePool {
    sealed trait State
  }

  trait CommandQueuePool extends OpenCL {
    // TODO: write buffer
    // TODO: read buffer

    // TODO: enqueue kernel
    // TODO: TDD

    protected val numberOfCommandQueuesForDevice: (Long, CLCapabilities) => Int

    @transient private lazy val commandQueues: Seq[Long] = {
      deviceIds.flatMap { deviceId =>
        val capabilities = deviceCapabilities(deviceId)
        for (i <- 0 until numberOfCommandQueuesForDevice(deviceId, capabilities)) yield {
          val supportedProperties = deviceLongInfo(deviceId, CL_DEVICE_QUEUE_PROPERTIES)
          val properties = Map(
            CL_QUEUE_PROPERTIES -> (supportedProperties & CL_QUEUE_ON_DEVICE)
          )
          createCommandQueue(deviceId, properties)
        }
      }
    }

    @transient
    lazy val Resource(acquireCommandQueue, shutdownCommandQueues) = AsynchronousPool.preloaded(commandQueues)

    private def deviceLongInfo(deviceId: Long, paramName: Int): Long = {
      val buffer = Array[Long](0L)

      println("call clGetDeviceInfo"); checkErrorCode(clGetDeviceInfo(deviceId, paramName, buffer, null))
      val Array(value) = buffer
      value
    }

    override def monadicClose: UnitContinuation[Unit] = {
      shutdownCommandQueues >> {
        for (commandQueue <- commandQueues) {

          println("call clReleaseCommandQueue"); checkErrorCode(clReleaseCommandQueue(commandQueue))
        }
        super.monadicClose
      }
    }
  }
  object Event {
    private[OpenCL] val eventCallback: CLEventCallback = CLEventCallback.create(new CLEventCallbackI {
      final def invoke(event: Long, status: Int, userData: Long): Unit = {
        val scalaCallback = try { memGlobalRefToObject[Int => Unit](userData) } finally {
          JNINativeInterface.DeleteGlobalRef(userData)
        }
        scalaCallback(status)
      }
    })
  }

  final case class Event[Owner <: Singleton with OpenCL](handle: Long)
      extends AnyVal
      with MonadicCloseable[UnitContinuation] {
    type Status = Int
    def waitForStatus(callbackType: Status): UnitContinuation[Status] = UnitContinuation.async { (continue: Status => Unit) =>
      val userData = JNINativeInterface.NewGlobalRef(continue)
      try {

        println("call clSetEventCallback");
        checkErrorCode(
          clSetEventCallback(
            handle,
            callbackType,
            Event.eventCallback,
            userData
          )
        )
      } catch {
        case NonFatal(e) =>
          //JNINativeInterface.DeleteGlobalRef(userData)
          throw e
      }
    }

    def waitFor(callbackType: Status): Future[Unit] = {
      println("waitFor")
      Future(TryT(waitForStatus(callbackType).map { status =>
        println("waitFor done")
        status match {
          case `callbackType` => Success(())
          case errorCode if errorCode < 0 =>
            println("Failure")
            Failure(Exceptions.fromErrorCode(errorCode))
          case status =>
            println("Failure status")
            throw new IllegalStateException(raw"""Invalid event status $status""")
        }
      }))
    }

    def waitForComplete(): Future[Unit] = waitFor(CL_COMPLETE)

    override def monadicClose: UnitContinuation[Unit] = {
      UnitContinuation.delay {
        Thread.sleep(10L)
        println("call clReleaseEvent");
        checkErrorCode(clReleaseEvent(handle))
      }
    }
  }

  object DeviceBuffer {
//    private[OpenCL] def delay[Owner <: Singleton with OpenCL, Element](
//        handle: => Long): Do[DeviceBuffer[Owner, Element]] = {
//      val bufferContinuation = UnitContinuation.delay {
//        Resource(
//          value = Success(DeviceBuffer[Owner, Element](handle)),
//          release = UnitContinuation.delay {
//
//            println(s"call clReleaseMemObject  (number of buffers = ${numberOfBuffers.decrementAndGet()})");
//            checkErrorCode(clReleaseMemObject(handle))
//          }
//        )
//      }
//      Do(TryT(ResourceT(bufferContinuation)))
//    }

    implicit def bufferBox[Owner <: Singleton with OpenCL, Element]: Box.Aux[DeviceBuffer[Owner, Element], Pointer] =
      new Box[DeviceBuffer[Owner, Element]] {
        override type Raw = Pointer

        override def box(raw: Raw): DeviceBuffer[Owner, Element] =
          new DeviceBuffer[Owner, Element](raw.address())

        override def unbox(boxed: DeviceBuffer[Owner, Element]): Raw = new Pointer.Default(boxed.handle) {}
      }

  }

  /** A [[https://www.khronos.org/registry/OpenCL/sdk/2.1/docs/man/xhtml/abstractDataTypes.html cl_mem]]
    * whose [[org.lwjgl.opencl.CL10.CL_MEM_TYPE CL_MEM_TYPE]] is buffer [[org.lwjgl.opencl.CL10.CL_MEM_OBJECT_BUFFER CL_MEM_OBJECT_BUFFER]].
    * @param handle The underlying `cl_mem`.
    * @note comment out extends AnyVal in case of https://github.com/scala/bug/issues/10647
    */
  final case class DeviceBuffer[Owner <: OpenCL with Singleton, Element](handle: Long) /* extends AnyVal */
      extends MonadicCloseable[UnitContinuation] {
    deviceBuffer =>

    override def monadicClose: UnitContinuation[Unit] = UnitContinuation.delay {
      println(s"call clReleaseMemObject  (number of buffers = ${numberOfBuffers.decrementAndGet()})");
      checkErrorCode(clReleaseMemObject(handle))
    }

    def slice(offset: Int, size: Int)(implicit
                                      memory: Memory[Element]): Do[DeviceBuffer[Owner, Element]] = {

      Do.monadicCloseable {
        val stack = stackPush()
        try {
          val errorCode = stack.ints(0)
          val region = CLBufferRegion.mallocStack(stack)
          region.set(offset.toLong * memory.numberOfBytesPerElement, size.toLong * memory.numberOfBytesPerElement)
          println("call clCreateSubBuffer");
          val newHandle = nclCreateSubBuffer(handle,
                                             CL_MEM_READ_WRITE,
                                             CL_BUFFER_CREATE_TYPE_REGION,
                                             region.address(),
                                             memAddress(errorCode))
          checkErrorCode(errorCode.get(0))
          DeviceBuffer[Owner, Element](newHandle)
        } finally {
          stack.close()
        }
      }
    }

    def numberOfBytes: Int = {
      val sizeBuffer: Array[Long] = Array(0L)

      println("call clGetMemObjectInfo"); checkErrorCode(clGetMemObjectInfo(handle, CL_MEM_SIZE, sizeBuffer, null))
      val Array(value) = sizeBuffer
      if (value.isValidInt) {
        value.toInt
      } else {
        throw new IllegalStateException(s"Buffer's numberOfBytes($value) is too large")
      }
    }

    def length(implicit memory: Memory[Element]): Int = numberOfBytes / memory.numberOfBytesPerElement

    private def enqueueReadBuffer[Destination](hostBuffer: Destination, preconditionEvents: Event[Owner]*)(
        implicit
        witnessOwner: Witness.Aux[Owner],
        memory: Memory.Aux[Element, Destination]): Do[Event[Owner]] = {

      println("acquireCommandQueue")
      witnessOwner.value.acquireCommandQueue.flatMap { commandQueue =>
        println("acquireCommandQueue done")
        Do.monadicCloseable {
          val outputEvent = {
            val stack = stackPush()
            try {
              val (inputEventBufferSize, inputEventBufferAddress) = if (preconditionEvents.isEmpty) {
                (0, NULL)
              } else {
                val inputEventBuffer = stack.pointers(preconditionEvents.view.map(_.handle): _*)
                (preconditionEvents.length, inputEventBuffer.address())
              }
              val outputEventBuffer = stack.pointers(0L)
              println("call clEnqueueReadBuffer");
              checkErrorCode(
                nclEnqueueReadBuffer(
                  commandQueue,
                  deviceBuffer.handle,
                  CL_FALSE,
                  0,
                  memory.remainingBytes(hostBuffer),
                  memory.address(hostBuffer),
                  inputEventBufferSize,
                  inputEventBufferAddress,
                  outputEventBuffer.address()
                ))
              Event[Owner](outputEventBuffer.get(0))
            } finally {
              stack.close()
            }
          }

          println("call clFlush"); checkErrorCode(clFlush(commandQueue))
          outputEvent
        }
      }
    }

    /** Returns an asynchronous operation of a buffer on host.
      *
      * The buffer may be [[java.nio.FloatBuffer FloatBuffer]],
      * [[java.nio.DoubleBuffer DoubleBuffer]]
      * or other buffer types according to [[Element]].
      *
      * @note The buffer is allocated by lwjgl, not JRE.
      *       As a result, you can only use the buffer inside a `map` or `flatMap` block,
      *       then it will be released by [[com.thoughtworks.raii.asynchronous.do.Do Do]] automatically.
      *       Assigning the buffer to another variable used outside `map` or `flatMap` block
      *       will cause memory access error.
      *
      */
    final def toHostBuffer(implicit witnessOwner: Witness.Aux[Owner],
                           memory: Memory[Element]): Do[memory.HostBuffer] = {
      Do(TryT(ResourceT(UnitContinuation.delay {
        val hostBuffer = memory.allocate(length)
        Resource(value = Success(hostBuffer), release = UnitContinuation.delay { memory.free(hostBuffer) })
      }))).flatMap { hostBuffer =>
        enqueueReadBuffer[memory.HostBuffer](hostBuffer)(witnessOwner, memory)
          .flatMap { event =>
            Do.garbageCollected(event.waitForComplete())
          }
          .intransitiveMap { _: Unit =>
            hostBuffer
          }
      }
    }
  }

  private[compute] final case class Kernel[Owner <: OpenCL with Singleton](handle: Long)
      extends AnyVal
      with MonadicCloseable[UnitContinuation] {


    def update[A](argIndex: Int, a: A)(implicit memory: Memory[A], witness: Witness.Aux[Owner]): Unit = {
//      if (functionName.startsWith("backward_data0_matrix_multiply")) {
//        argIndex match {
//          case 0 | 1 | 2 =>
//            val buffer = a.asInstanceOf[DeviceBuffer[Owner, Float]]
////            buffer.toHostBuffer.map(Memory.FloatMemory.toArray).run.onComplete {
////              case Success(array) =>
////                println(s"$handle[$argIndex] = ${array.toSeq}")
////              case Failure(e) =>
////                e.printStackTrace
////            }
//          case _ =>
//        }
//
//      }
      println(s"index = $argIndex, a = $a, handle = $handle")

      val stack = stackPush()

      try {

        val byteBuffer = stack.malloc(memory.numberOfBytesPerElement)
        memory.put(memory.fromByteBuffer(byteBuffer), 0, a)
        println("call clSetKernelArg");
        checkErrorCode(nclSetKernelArg(handle, argIndex, byteBuffer.remaining, memAddress(byteBuffer)))
      } finally {
        stack.close()
      }

    }

    def functionName: String = {
      val stack = stackPush()

      try {

        val functionNameSizePointer = stack.mallocPointer(1)

        println("call clGetKernelInfo");
        checkErrorCode(
          clGetKernelInfo(this.handle, CL_KERNEL_FUNCTION_NAME, null: PointerBuffer, functionNameSizePointer))
        val functionNameSize = functionNameSizePointer.get(0).toInt
        val functionNameBuffer = stack.malloc(functionNameSize)

        println("call clGetKernelInfo");
        checkErrorCode(
          clGetKernelInfo(this.handle, CL_KERNEL_FUNCTION_NAME, functionNameBuffer, functionNameSizePointer))
        decodeString(functionNameBuffer)
      } finally {
        stack.close()
      }
    }
    
    def enqueue(globalWorkSize: Long*)(implicit witnessOwner: Witness.Aux[Owner]): Do[Event[Owner]] = {
      println(s"globalWorkSize=${globalWorkSize} handle = $handle")
      witnessOwner.value.acquireCommandQueue.flatMap { commandQueue =>
        Do.monadicCloseable {
          val stack = stackPush()
          val outputEvent = try {
            val inputEventBuffer = null
            val outputEventBuffer = stack.pointers(0L)
            println(s"clEnqueueNDRangeKernel ${functionName} globalWorkSize=${globalWorkSize} handle = $handle thread id = ${Thread.currentThread().getId}");

            println("call clEnqueueNDRangeKernel");
            checkErrorCode(
              clEnqueueNDRangeKernel(
                commandQueue,
                handle,
                globalWorkSize.size,
                null,
                stack.pointers(globalWorkSize: _*),
                null,
                inputEventBuffer,
                outputEventBuffer
              ))
            Event[Owner](outputEventBuffer.get(0))
          } finally {
            stack.close()
          }

          println("call clFlush"); checkErrorCode(clFlush(commandQueue))
          outputEvent
        }
      }
    }

    override def monadicClose: UnitContinuation[Unit] = {
      UnitContinuation.delay {

        println("call clReleaseKernel"); checkErrorCode(clReleaseKernel(handle))
      }
    }
  }

  private[compute] final case class Program[Owner <: OpenCL with Singleton](handle: Long)
      extends AnyVal
      with MonadicCloseable[UnitContinuation] {

    private def numberOfKernels: Int = {
      val result = Array.ofDim[Int](1)

      println("call clCreateKernelsInProgram"); checkErrorCode(clCreateKernelsInProgram(handle, null, result))
      result(0)
    }

    def deviceIds: Seq[Long] = {
      val stack = stackPush()
      try {
        val sizeBuffer = stack.mallocPointer(1)

        println("call clGetProgramInfo");
        checkErrorCode(clGetProgramInfo(this.handle, CL_PROGRAM_DEVICES, null: PointerBuffer, sizeBuffer))
        val numberOfDeviceIds = sizeBuffer.get(0).toInt / POINTER_SIZE
        val programDevicesBuffer = stack.mallocPointer(numberOfDeviceIds)

        println("call clGetProgramInfo");
        checkErrorCode(clGetProgramInfo(this.handle, CL_PROGRAM_DEVICES, programDevicesBuffer, sizeBuffer))
        (0 until numberOfDeviceIds).map(programDevicesBuffer.get)
      } finally {
        stack.close()
      }
    }

    def createKernels(): Seq[Kernel[Owner]] = {
      (0 until createKernelBuffer().capacity).map { i =>
        Kernel[Owner](createKernelBuffer().get(i))
      }
    }

    private def createKernelBuffer(): PointerBuffer = {
      val kernelBuffer = BufferUtils.createPointerBuffer(numberOfKernels)

      println("call clCreateKernelsInProgram");
      checkErrorCode(clCreateKernelsInProgram(handle, kernelBuffer, null: IntBuffer))
      kernelBuffer
    }

    def createFirstKernel(): Kernel[Owner] = {
      val stack = stackPush()
      try {
        val kernelBuffer = stack.mallocPointer(1)

        println("call clCreateKernelsInProgram");
        checkErrorCode(clCreateKernelsInProgram(handle, kernelBuffer, null: IntBuffer))
        Kernel(kernelBuffer.get(0))
      } finally {
        stack.close()
      }
    }

    private def buildLogs(deviceIds: Seq[Long]): Map[Long /* device ID */, String] = {
      val stack = stackPush()
      try {
        val sizeBuffer = stack.mallocPointer(1)
        deviceIds.view.map { deviceId =>
          println("call clGetProgramBuildInfo");
          checkErrorCode(
            clGetProgramBuildInfo(this.handle, deviceId, CL_PROGRAM_BUILD_LOG, null: PointerBuffer, sizeBuffer))
          val logBuffer = stack.malloc(sizeBuffer.get(0).toInt)

          println("call clGetProgramBuildInfo");
          checkErrorCode(clGetProgramBuildInfo(this.handle, deviceId, CL_PROGRAM_BUILD_LOG, logBuffer, sizeBuffer))
          (deviceId, decodeString(logBuffer))
        }.toMap
      } finally {
        stack.close()
      }
    }

    private def checkBuildErrorCode(deviceIdsOption: Option[Seq[Long]], errorCode: Int): Unit = {
      errorCode match {
        case CL_BUILD_PROGRAM_FAILURE =>
          val logs = deviceIdsOption match {
            case None      => buildLogs(this.deviceIds)
            case Some(ids) => buildLogs(ids)
          }
          throw new Exceptions.BuildProgramFailure(logs)
        case _ => checkErrorCode(errorCode)
      }
    }

    def build(deviceIds: Seq[Long], options: CharSequence = ""): Unit = {
      val stack = stackPush()
      try {

        println("call clBuildProgram");
        checkBuildErrorCode(Some(deviceIds), clBuildProgram(handle, stack.pointers(deviceIds: _*), options, null, NULL))
      } finally {
        stack.close()
      }
    }

    def build(options: CharSequence): Unit = {

      println("call clBuildProgram"); checkBuildErrorCode(None, clBuildProgram(handle, null, options, null, NULL))
    }

    def build(): Unit = build("")

    def monadicClose = UnitContinuation.delay {

      println("call clReleaseProgram"); OpenCL.checkErrorCode(clReleaseProgram(handle))
    }
  }

  object Program {

    override def finalize(): Unit = {
      programCallback.close()
    }

    val programCallback = CLProgramCallback.create(new CLProgramCallbackI {
      override def invoke(program: Long, userData: Long): Unit = {
        val scalaCallback = try {
          memGlobalRefToObject[Unit => Unit](userData)
        } finally {
          JNINativeInterface.DeleteGlobalRef(userData)
        }
        scalaCallback(())
      }
    })

  }

}

trait OpenCL extends MonadicCloseable[UnitContinuation] with ImplicitsSingleton {
  type Program = OpenCL.Program[this.type]
  type Event = OpenCL.Event[this.type]
  protected def createProgramWithSource(sourceCode: TraversableOnce[CharSequence]): Program = {
    val stack = stackPush()
    try {
      val errorCodeBuffer = stack.ints(CL_SUCCESS)
      val codeBuffers = (for {
        snippet <- sourceCode
        if snippet.length > 0
      } yield memUTF8(snippet, false)).toArray
      val pointers = memAllocPointer(codeBuffers.length)
      val lengths = memAllocPointer(codeBuffers.length)
      for (buffer <- codeBuffers) {
        pointers.put(buffer)
        lengths.put(buffer.remaining)
      }
      pointers.position(0)
      lengths.position(0)
      try {

        println("call clCreateProgramWithSource");
        val programHandle = clCreateProgramWithSource(context, pointers, lengths, errorCodeBuffer)
        checkErrorCode(errorCodeBuffer.get(0))
        new Program(programHandle)
      } finally {
        memFree(pointers)
        memFree(lengths)
        codeBuffers.foreach(memFree)
      }
    } finally {
      stack.close()
    }
  }

  protected def acquireCommandQueue: Do[Long]

  def monadicClose: UnitContinuation[Unit] = UnitContinuation.delay {

    println("call clReleaseContext"); checkErrorCode(clReleaseContext(context))
  }

  protected def handleOpenCLNotification(errorInfo: String, privateInfo: ByteBuffer): Unit

  import OpenCL._

  protected val platformId: Long
  protected val deviceIds: Seq[Long]

  @transient
  private lazy val platformCapabilities: CLCapabilities = {
    CL.createPlatformCapabilities(platformId)
  }

  protected def createCommandQueue(deviceId: Long, properties: Map[Int, Long]): Long = {
    if (deviceCapabilities(deviceId).OpenCL20) {
      val cl20Properties = (properties.view.flatMap { case (key, value) => Seq(key, value) } ++ Seq(0L)).toArray
      val a = Array(0)

      println("call clCreateCommandQueueWithProperties");
      val commandQueue =
        clCreateCommandQueueWithProperties(platformId, deviceId, cl20Properties, a)
      checkErrorCode(a(0))
      commandQueue
    } else {
      val cl10Properties = properties.getOrElse(CL_QUEUE_PROPERTIES, 0L)
      val a = Array(0)

      println("call clCreateCommandQueue");
      val commandQueue = clCreateCommandQueue(context, deviceId, cl10Properties, a)
      checkErrorCode(a(0))
      commandQueue
    }
  }

  @transient
  protected lazy val deviceCapabilities: Long => CLCapabilities = {
    Memo.mutableMapMemo(new ConcurrentHashMap[Long, CLCapabilities].asScala) { deviceId =>
      CL.createDeviceCapabilities(deviceId, platformCapabilities)
    }
  }

  @transient
  protected lazy val context: Long = {
    val stack = stackPush()
    try {
      val errorCodeBuffer = stack.ints(CL_SUCCESS)
      val contextProperties = stack.pointers(CL_CONTEXT_PLATFORM, platformId, 0)
      val deviceIdBuffer = stack.pointers(deviceIds: _*)

      println("call clCreateContext");
      val context =
        clCreateContext(contextProperties,
                        deviceIdBuffer,
                        OpenCL.contextCallback,
                        JNINativeInterface.NewWeakGlobalRef(this),
                        errorCodeBuffer)
      checkErrorCode(errorCodeBuffer.get(0))
      context
    } finally {
      stack.close()
    }
  }
  trait ImplicitsApi {}
  type Implicits <: ImplicitsApi

  val implicits: Implicits

  type DeviceBuffer[Element] = OpenCL.DeviceBuffer[this.type, Element]

  /** Returns an uninitialized buffer of `Element` on device.
    */
  def allocateBuffer[Element](size: Long)(implicit memory: Memory[Element]): Do[DeviceBuffer[Element]] =
    Do.monadicCloseable {
      val stack = stackPush()
      try {
        val errorCodeBuffer = stack.ints(CL_SUCCESS)

        println(s"call clCreateBuffer (number of buffers = ${numberOfBuffers.incrementAndGet()})");
        val buffer =
          clCreateBuffer(context, CL_MEM_READ_WRITE, memory.numberOfBytesPerElement * size, errorCodeBuffer)
        checkErrorCode(errorCodeBuffer.get(0))
        DeviceBuffer[this.type, Element](buffer)
      } finally {
        stack.pop()
      }
    }

  /** Returns a buffer of `Element` on device whose content is copied from `hostBuffer`.
    */
  def allocateBufferFrom[Element, HostBuffer](hostBuffer: HostBuffer)(
      implicit memory: Memory.Aux[Element, HostBuffer]): Do[DeviceBuffer[Element]] =
    Do.monadicCloseable {
      val stack = stackPush()
      try {
        val errorCodeBuffer = stack.ints(CL_SUCCESS)
        println(s"call clCreateBuffer (number of buffers = ${numberOfBuffers.incrementAndGet()})");
        val buffer = nclCreateBuffer(context,
                                     CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE,
                                     memory.remainingBytes(hostBuffer),
                                     memory.address(hostBuffer),
                                     memAddress(errorCodeBuffer))
        checkErrorCode(errorCodeBuffer.get(0))
        DeviceBuffer[this.type, Element](buffer)
      } finally {
        stack.pop()
      }
    }

}
