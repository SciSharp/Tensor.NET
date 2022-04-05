using Numnet.Native;
using System.Text;
using Numnet.Exceptions;

namespace Numnet.Tensor.Base{
    public class TensorShape{
        public static readonly int MAX_NDIM = 4;
        public int[] Shape { get; internal set; } = new int[MAX_NDIM];
        public int NDim{ get; internal set; }
        public TensorShape(params int[] shape){
            if(shape.Length > MAX_NDIM){
                throw new DimExceedException(shape.Length);
            }
            NDim = shape.Length;
            for (int i = 0; i < NDim; i++){
                Shape[i] = shape[NDim - i - 1];
            }
        }
        public TensorShape(Span<int> shape){
            if(shape.Length > MAX_NDIM){
                throw new DimExceedException(shape.Length);
            }
            NDim = shape.Length;
            for (int i = 0; i < NDim; i++){
                Shape[i] = shape[NDim - i - 1];
            }
        }
        public TensorShape(TensorShape rhs){
            NDim = rhs.NDim;
            rhs.Shape.CopyTo(Shape.AsSpan());
        }
    }
    public sealed class TensorLayout:TensorShape
    {
        public DType DType { get; internal set; }
        public int Offset{ get; internal set; }
        public int[] Stride { get; internal set; } = new int[MAX_NDIM];
        public TensorLayout()
        {
            DType = DType.Invalid;
            NDim = 0;
            Offset = 0;
        }
        public TensorLayout(DType dtype, Span<int> shape):base(shape)
        {
            DType = dtype;
            Offset = 0;
            InitContiguousLayout();
        }

        public TensorLayout(DType dtype, int[] shape):base(shape)
        {
            DType = dtype;
            Offset = 0;
            InitContiguousLayout();
        }

        public TensorLayout(DType dtype, TensorShape shape):base(shape)
        {
            DType = dtype;
            Offset = 0;
            InitContiguousLayout();
        }

        public TensorLayout(TensorLayout rhs){
            DType = rhs.DType;
            Offset = rhs.Offset;
            NDim = rhs.NDim;
            rhs.Shape.CopyTo(Shape.AsSpan());
            rhs.Stride.CopyTo(Stride.AsSpan());
        }

        public int TotalElemCount(){
            if(NDim == 0){
                return 0;
            }
            int res = 1;
            for (int i = 0; i < NDim; i++){
                res *= Shape[i];
            }
            return res;
        }

        public bool IsScalar(){
            return NDim == 1 && Shape[0] == 1;
        }

        public override string ToString()
        {
            return $"TensorLayout({GetInfoString()})";
        }

        internal string GetInfoString(){
            StringBuilder r = new StringBuilder();
            if (NDim == 0) {
                r.Append(" Scalar");
            } else {
                r.Append("shape = {");
                for (int i = 0; i < NDim; i++) {
                    r.Append(Shape[NDim - 1 - i]);
                    if (i != NDim - 1) r.Append(", ");
                }
                r.Append("}");
            }
            r.Append(", dtype = ");
            r.Append(DType);
            return r.ToString();
        }

        internal void InitContiguousLayout()
        {
            int s = 1;
            for (int i = 0; i < NDim; i++)
            {
                Stride[i] = s;
                s *= Shape[i];
            }
        }
        internal TensorLayout Reshape(TensorShape targetShape, bool isImage){
            int targetNDim = targetShape.NDim;
            if(targetNDim <= 0){
                throw new InvalidShapeException(targetShape.Shape, "Reshape");
            }
            // bool isEmptyShape = false;
            int targetTotalElems = 1;
            for (int i = 0; i < targetNDim; ++i) {
                // if (targetShape[i] <= 0) {
                //     isEmptyShape = true;
                //     break;
                // }
                targetTotalElems *= targetShape.Shape[i];
            }

            if(this.TotalElemCount() != targetTotalElems){
                throw new InvalidShapeException($"Number of elements does not match in reshape: src = {this.TotalElemCount()}, dst = {targetTotalElems}.");
            }

            TensorLayout res = new TensorLayout(this.DType, targetShape);
            // Maybe the process here is not correct when dealing with image.
            // Because the shape is not converted into contiguous before the process.
            if (isImage) {
                if(this.NDim < 2){
                    throw new InvalidOperationException("The tensor to reshape is not an image.");
                }
                else if(targetNDim < 2){
                    throw new InvalidShapeException($"The target shape({string.Join(',', targetShape)}) is not an invalid shape for image.");
                }
                else if(this.Shape[0] != targetShape.Shape[1] || this.Shape[1] != targetShape.Shape[0]){
                    throw new InvalidShapeException($"The target shape({string.Join(',', targetShape)}) does not match the current shape({string.Join(',', this.Shape)}) as an image.");
                }
                for (int i = 2; i < this.NDim; i++) {
                    if (targetNDim > i && targetShape.Shape[i] != this.Shape[i]) {
                        throw new InvalidShapeException($"The target shape({string.Join(',', targetShape)}) does not match the current shape({string.Join(',', this.Shape)}) as an image to reshape.");
                    }
                }
                (res.Stride[0], res.Stride[1]) = (res.Stride[1], res.Stride[0]);
                return res;
            }

            // if (isEmptyShape) {
            //     return res;
            // }
            // var cont = CollapseContiguous();

            // int sdim = 0, prod = 1, cont_sdim = 0;
            // for (int i = targetNDim; i >= 0; i--) {
            //     if(cont_sdim >= cont.NDim){
            //         throw new InvalidShapeException($"The target shape({string.Join(',', targetShape)}) does not match the current shape({string.Join(',', this.Shape)}) to reshape.");
            //     }
            //     prod *= res.Shape[i];
            //     if (prod > cont.Shape[cont_sdim])
            //         throw new InvalidShapeException($"The target shape({string.Join(',', targetShape)}) does not match the current shape({string.Join(',', this.Shape)}) to reshape.");

            //     if (prod == cont.Shape[cont_sdim] &&
            //         (i + 1 >= targetNDim || targetShape[i + 1] != 1)) {
            //         int s = cont.Stride[cont_sdim];
            //         for (int j = i; j >= sdim; j--) {
            //             res.Stride[j] = s;
            //             s *= res.Shape[j];
            //         }
            //         cont_sdim++;
            //         sdim = i + 1;
            //         prod = 1;
            //     }
            // }

            return res;
        }

        internal TensorLayout CollapseContiguous() {
            if(NDim == 0){
                throw new InvalidShapeException($"The ndim of the tensor that try to collapse contiguously is 0.");
            }
            TensorLayout res = new TensorLayout(this);

            // remove all dims with shape 1
            for (int i = 0; i <= res.NDim - 1 && res.NDim >= 2; ++i) {
                if (res.Shape[i] <= 0) {
                    // empty tensor
                    res.NDim = 1;
                    res.Shape[0] = 0;
                    res.Stride[0] = 1;
                    return res;
                }
                if (res.Shape[i] == 1) res.RemoveAxisInplace(i);
            }

            if (res.NDim == 1) {
                if (res.Shape[0] <= 1) {
                    // make it the "most canonical" contiguous layout for scalars or
                    // empty tensors
                    res.Stride[0] = 1;
                }
                return res;
            }

            if(res.NDim <= 0 || res.Shape[res.NDim - 1] <= 0){
                throw new InvalidShapeException(this.Shape, "CollapseContiguous");
            }
            for (int i = 1; i <= res.NDim - 1; i++) {
                if(res.Shape[i] <= 0){
                    throw new InvalidShapeException(this.Shape, "CollapseContiguous");
                }
                if (res.Stride[i] == res.Stride[i - 1] * res.Shape[i - 1]) {
                    res.Shape[i] *= res.Shape[i - 1];
                    res.Stride[i] = res.Stride[i - 1];
                    res.RemoveAxisInplace(i - 1);
                }
            }
            return res;
        }

        internal void RemoveAxisInplace(int axis) {
            if(NDim < 2){
                throw new InvalidShapeException($"Could not remove axis of a tensor with only {NDim} dims.");
            }
            else if(axis < NDim){
                throw new InvalidArgumentException($"Axis to remove exceeds the NDim. Axis is {axis}, NDim is {NDim}.");
            }
            NDim--;
            for (int i = axis; i < NDim; ++i) {
                Shape[i] = Shape[i + 1];
                Stride[i] = Stride[i + 1];
            }
        }

        internal void BroadcastInplace(TensorShape targetShape){
            int targetNDim = targetShape.NDim;
            if(NDim <= 0 || targetNDim <= 0){
                throw new InvalidShapeException("Cannot broadcast (to) empty tensor shape");
            }

            if (IsScalar()) {
                NDim = targetNDim;
                for (int i = 0; i < targetNDim; i++) {
                    Shape[i] = targetShape.Shape[i];
                    Stride[i] = targetShape.Shape[i] == 1?targetShape.Shape[i]:0;
                }
                return;
            }

            if(targetNDim < NDim){
                throw new InvalidShapeException($"Dimension after broadcast is less than that before braodcast. " +
                        $"The src shape is ({string.Join(',', Shape.AsSpan(0, NDim).ToArray())}), dst shape is ({string.Join(',', targetShape)}).");
            }

            for (int i = 0; i < targetNDim; ++i) {
                int cur_shape = (i < NDim ? Shape[i] : 1), cur_stride = (i < NDim ? Stride[i] : 0);
                if (targetShape.Shape[i] != cur_shape) {
                    if(cur_shape != 1 && cur_stride != 0){
                        throw new InvalidShapeException("Broadcast on dim with shape not equal to 1: " + 
                            $"src_shape=({string.Join(',', Shape.AsSpan(0, NDim).ToArray())}) dst_shape=({string.Join(',', targetShape)})");
                    }
                    Shape[i] = targetShape.Shape[i];
                    Stride[i] = 0;
                } else {
                    Shape[i] = cur_shape;
                    Stride[i] = cur_stride;
                }
            }
            NDim = targetNDim;
        }

        internal TensorLayout BroadcastTo(TensorShape targetShape){
            TensorLayout res = new TensorLayout(this);
            res.BroadcastInplace(targetShape);
            return res;
        }
    }
    

}