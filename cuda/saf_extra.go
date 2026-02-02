package cuda

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#cgo LDFLAGS: -L${SRCDIR} -lsaf_wrapper -L/usr/local/cuda/lib64 -lcudart -lcurand

#include "saf_wrapper_cu.h"
#include <stdlib.h>
*/
import "C"

import (
	"unsafe"

	"github.com/mumax/3/data"
)

// SAFSetValue_CUDA helper defined in a new file to avoid build caching issues
func SAFSetValue_CUDA_Fixed(dst *data.Slice, val float32) {
	N := dst.Len()
	C.k_setValue_async(
		(*C.float)(unsafe.Pointer(dst.DevPtr(0))),
		C.float(val),
		C.int(N),
		nil)
}
