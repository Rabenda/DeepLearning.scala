package com.thoughtworks.compute

import java.io.Closeable

object Closeables {

  trait IsClosed {
    protected final var closed = false

  }

  trait AssertionFinalizer { this: IsClosed =>
    override protected final def finalize(): Unit = {
      if (!closed) {
        throw new IllegalStateException("close() must be called before garbage collection.")
      }
    }
  }

  trait IdempotentFinalizer { this: Closeable =>
    override protected final def finalize(): Unit = close()
  }

  /**
    * An idempotent `Closeable`.
    * @author 杨博 (Yang Bo) &lt;pop.atry@gmail.com&gt;
    */
  trait IdempotentCloseable extends Closeable {

    protected def forceClose(): Unit

    protected final var closed = false

    /**
      * Calls [[forceClose]] and then marks this [[IdempotentCloseable]] as closed if this [[IdempotentCloseable]] was not closed; does nothing otherwise.
      */
    override final def close(): Unit = {
      val wasClosed = synchronized {
        val wasClosed = closed
        if (!wasClosed) {
          closed = true
        }
        wasClosed
      }
      if (!wasClosed) {
        forceClose()
      }
    }

  }

  trait AssertionAutoCloseable extends AutoCloseable with IsClosed {

    protected def forceClose(): Unit

    /**
      * Calls [[forceClose]] and then marks this [[AssertionAutoCloseable]] as closed if this [[AssertionAutoCloseable]] was not closed; throw an exception otherwise.
      */
    @throws(classOf[IllegalStateException])
    override final def close(): Unit = {
      val wasClosed = synchronized {
        val wasClosed = closed
        if (!wasClosed) {
          closed = true
        }
        wasClosed
      }
      if (wasClosed) {
        throw new IllegalStateException("Can't close more than once.")
      } else {
        forceClose()
      }
    }

  }

}
