package com.thoughtworks.deeplearning.plugins

import java.nio.ByteBuffer
import java.util.concurrent.ForkJoinPool

import com.thoughtworks.compute.{Memory, OpenCL}
import com.thoughtworks.compute.OpenCL._
import com.thoughtworks.continuation.UnitContinuation
import com.thoughtworks.future.{Future, ParallelFuture}
import com.thoughtworks.raii.asynchronous._
import com.thoughtworks.future._
import com.thoughtworks.continuation._
import com.thoughtworks.feature.Factory
import com.thoughtworks.feature.mixins.ImplicitsSingleton
import org.lwjgl.opencl.CLCapabilities
import shapeless.Witness

import scala.concurrent.ExecutionContext
import scalaz.Semigroup
import scalaz.std.stream._
import scalaz.syntax.all._
import scalaz.syntax.tag._
import scalaz.Tags.Parallel

object ExitCode134MatrixMultiply {

  implicit val executionContext = ExecutionContext.fromExecutorService(new ForkJoinPool(50))

  trait TestKernels extends OpenCL with OpenCL.CommandQueuePool {

    private implicit def witnessSelf: Witness.Aux[this.type] = Witness.mkWitness(this)

    private lazy val matrixMultiplyProgram: Program = {
      val program = createProgramWithSource(
        Seq("""
				kernel void matrix_multiply(
					global const float* /*restrict*/ input0,
					global const float* /*restrict*/ input1,
					global float* restrict output,
					size_t matrix0_columns,
					size_t matrix1_columns
				) {
					const size_t i = get_global_id(0);
					const size_t j = get_global_id(1);

					float value = 0.0f;
					for (int k = 0; k < matrix0_columns; ++k) {
						float elementA = input0[i * matrix0_columns + k];
						float elementB = input1[k * matrix1_columns + j];
						value += elementA * elementB;
					}
					output[i * matrix1_columns + j] = value;
				}
			""")
      )
      program.build()
      program
    }

    private lazy val backwardData1MatrixMultiplyProgram: Program = {
      val program = createProgramWithSource(
        Seq(
          """
					kernel void backward_data1_matrix_multiply(global const float* /*restrict*/ input0, global const float* /*restrict*/ input1, global float* restrict output, size_t matrix0_columns, size_t matrix1_columns) {
						const size_t i = get_global_id(0);
						const size_t j = get_global_id(1);

						float value = 0.0f;
						for (int k = 0; k < matrix0_columns; ++k) {
							float elementA = input0[k * matrix0_columns + i];
							float elementB = input1[k * matrix1_columns + j];
							value += elementA * elementB;
						}
						output[i * matrix0_columns + j] = value;
					}
				""")
      )

      // matrix0_columns == matrix1_rows
      program.build()
      program
    }

    private lazy val subtractInplaceProgram: Program = {
      val program = createProgramWithSource(
        Seq("""
        kernel void subtract_inplace(global float* /*restrict*/ input0, global const float* /*restrict*/ input1) {
          const size_t index = get_global_id(0);
          input0[index] -= input1[index];
        }
      """)
      )

      program.build()
      program
    }
//
//    override def monadicClose: UnitContinuation[Unit] = {
//	    backwardData1MatrixMultiplyProgram.monadicClose >> super.monadicClose
//    }

    def testSubtractInplace(output: DeviceBuffer[Float], updateData: DeviceBuffer[Float]): Do[DeviceBuffer[Float]] = {

      Do.monadicCloseable(subtractInplaceProgram.createFirstKernel)
        .flatMap { kernel =>
          kernel(0) = output
          kernel(1) = updateData
          kernel.enqueue(output.length).flatMap { event =>
            Do.garbageCollected(event.waitForComplete())
          }
        }
        .intransitiveMap { _: Unit =>
          output
        }

    }

    def test(input: DeviceBuffer[Float],
             weight: DeviceBuffer[Float],
             matrix0Rows: Int,
             matrix0Columns: Int,
             matrix1Rows: Int,
             matrix1Columns: Int): Do[DeviceBuffer[Float]] = {
      allocateBuffer[Float](matrix0Rows * matrix1Rows).flatMap { output: DeviceBuffer[Float] =>
        Do.monadicCloseable(backwardData1MatrixMultiplyProgram.createFirstKernel)
          .flatMap { kernel =>
            kernel(0) = input
            kernel(1) = weight
            kernel(2) = output
            kernel(3) = matrix0Columns
            kernel(4) = matrix1Columns
            kernel.enqueue(matrix0Rows, matrix1Rows).flatMap { event =>
              Do.garbageCollected(event.waitForComplete())
            }
          }
          .intransitiveMap { _: Unit =>
            output
          }
      }

    }
  }

  private val handleOpenCLNotification = { (errorInfo: String, buffer: ByteBuffer) =>
    if (buffer.remaining > 0) {
      val hexText = for (i <- (buffer.position until buffer.limit).view) yield {
        f"${buffer.get(i)}%02X"
      }
      Console.err.println(hexText.mkString(errorInfo, " ", ""))
      Console.err.flush()
    } else {
      Console.err.println(errorInfo)
      Console.err.flush()
    }
  }

  private[ExitCode134MatrixMultiply] trait DeviceBufferOf extends OpenCL {
    def deviceBufferOf[Element](elements: Element*)(implicit memory: Memory[Element]): Do[DeviceBuffer[Element]] = {
      val hostBuffer: memory.HostBuffer = memory.allocate(elements.length)
      // TODO: optimize the performance
      for ((element, i) <- elements.view.zipWithIndex) {
        memory.put(hostBuffer, i, element)
      }
      allocateBufferFrom[Element, memory.HostBuffer](hostBuffer)(memory)
    }
  }

  def main(array: Array[String]): Unit = {
    new ExitCode134MatrixMultiply().testMartixMultiply()
  }

}

class ExitCode134MatrixMultiply {
  import ExitCode134MatrixMultiply._
  def testMartixMultiply(): Unit = {

    val doHyperparameter = Do.monadicCloseable {
      Factory[
        TestKernels with OpenCL with OpenCL.UseAllGpuDevice with OpenCL.UseFirstPlatform with ImplicitsSingleton with OpenCL.CommandQueuePool with DeviceBufferOf]
        .newInstance(
          handleOpenCLNotification = handleOpenCLNotification,
          numberOfCommandQueuesForDevice = { (deviceId: Long, capabilities: CLCapabilities) =>
            1
          }
        )
    }
//
//	  def t(fa: Stream[Int])(Future[DeviceBuffer[Float]]): Stream[Future[DeviceBuffer[Float]]] = ???

    doHyperparameter
      .flatMap { hyperparameter2 =>
        val hyperparameter = hyperparameter2
        import hyperparameter._

        val total = 500
        deviceBufferOf(0f, 1f, 2f).flatMap { input =>
          deviceBufferOf(100f, 100f, 100f).flatMap { weight =>
            def train: Future[Unit] = {
              ((0 until total).toStream).traverseU_ { i: Int =>
                hyperparameter
                  .testSubtractInplace(
                    input,
                    weight,
                  )
                  .map { buffer =>
                    ()
                  }
                  .run
              }

            }
            Do.garbageCollected(train)
          }
        }
      }
      .run
      .blockingAwait

  }

}
