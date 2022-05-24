using Tensornet.Native;
using Tensornet.Exceptions;
using Tensornet.Native.Param;

namespace Tensornet{
    public enum PadMode : Int32{
        Constant = 1,
        Edge = 2,
        Maximum = 3,
        Minimum = 4,
        Medium = 5,
        Mean = 6
    }

    public static class PadExtension{
        /// <summary>
        /// Pad a tensor with given values.
        /// For details, please refer to https://numpy.org/doc/stable/reference/generated/numpy.pad.html?highlight=pad#numpy.pad
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"></param>
        /// <param name="widths"> Number of values padded to the edges of each axis. ((before_1, after_1), … (before_N, after_N)) unique pad widths for each axis. </param>
        /// <param name="mode"> The mode of the padding.</param>
        /// <param name="constants"> Used in ‘constant’. The values to set the padded values for each axis. ((before_1, after_1), ... (before_N, after_N)) unique pad constants for each axis. Default is 0.</param>
        /// <returns></returns>
        /// <exception cref="InvalidParamException"></exception>
        /// <exception cref="NotImplementedException"></exception>
        public static Tensor<T> Pad<T>(this Tensor<T> src, (int, int)[] widths, PadMode mode = PadMode.Constant, 
            (double, double)[]? constants = null) where T : struct, IEquatable<T>, IConvertible
        {
            if(widths.Length != src.TLayout.NDim){
                throw new InvalidParamException($"The tensor to pad has {src.TLayout.NDim} dims, but only {widths.Length} dims of width are specified.");
            }
            if(mode == PadMode.Medium || mode == PadMode.Mean){
                throw new NotImplementedException("This padding mode has not been implemented yet. It will be implemented in the future version.");
            }
            int[] paramWidths = new int[widths.Length * 2];
            for (int i = 0; i < widths.Length; i++){
                paramWidths[i * 2] = widths[i].Item1;
                paramWidths[i * 2 + 1] = widths[i].Item2;
            }
            Tensor<T> res = new Tensor<T>(DeduceLayout(src.TLayout, paramWidths));
            res.TLayout.InitContiguousLayout();
            if (mode == PadMode.Constant)
            {
                double[] paramConstants = new double[paramWidths.Length];
                if(constants is null){
                    paramConstants.AsSpan().Fill(0);
                }
                else{
                    if(constants.Length != widths.Length){
                        throw new InvalidParamException($"{widths.Length} pairs constant should be specified, but only get {constants.Length} pairs.");
                    }
                    for (int i = 0; i < constants.Length; i++){
                        paramConstants[i * 2] = constants[i].Item1;
                        paramConstants[i * 2 + 1] = constants[i].Item2;
                    }
                }
                PadInternal(src, res, mode, paramWidths, paramConstants);
            }
            else{
                PadInternal(src, res, mode, paramWidths, null);
            }
            return res;
        }
        /// <summary>
        /// Pad a tensor with given values.
        /// For details, please refer to https://numpy.org/doc/stable/reference/generated/numpy.pad.html?highlight=pad#numpy.pad
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"></param>
        /// <param name="widths"> Number of values padded to the edges of each axis. ((width_1, width_1), … (width_N, width_N)) same pad widths for each axis. </param>
        /// <param name="mode"> The mode of the padding.</param>
        /// <param name="constants"> Used in ‘constant’. The values to set the padded values for each axis. ((value_1, value_1), ... (value_N, value_N)) same pad constants for each axis. Default is 0.</param>
        /// <returns></returns>
        /// <exception cref="InvalidParamException"></exception>
        /// <exception cref="NotImplementedException"></exception>
        public static Tensor<T> Pad<T>(this Tensor<T> src, int[] widths, PadMode mode = PadMode.Constant, 
            double[]? constants = null) where T : struct, IEquatable<T>, IConvertible
        {
            if(widths.Length != src.TLayout.NDim){
                throw new InvalidParamException($"The tensor to pad has {src.TLayout.NDim} dims, but only {widths.Length} dims of width are specified.");
            }
            if(mode == PadMode.Medium || mode == PadMode.Mean){
                throw new NotImplementedException("This padding mode has not been implemented yet. It will be implemented in the future version.");
            }
            int[] paramWidths = new int[widths.Length * 2];
            for (int i = 0; i < widths.Length; i++){
                paramWidths[i * 2] = paramWidths[i * 2 + 1] = widths[i];
            }
            Tensor<T> res = new Tensor<T>(DeduceLayout(src.TLayout, paramWidths));
            res.TLayout.InitContiguousLayout();
            if (mode == PadMode.Constant)
            {
                double[] paramConstants = new double[paramWidths.Length];
                if(constants is null){
                    paramConstants.AsSpan().Fill(0);
                }
                else{
                    if(constants.Length != widths.Length){
                        throw new InvalidParamException($"{widths.Length} pairs constant should be specified, but only get {constants.Length} pairs.");
                    }
                    for (int i = 0; i < constants.Length; i++){
                        paramConstants[i * 2] = paramConstants[i * 2 + 1] = constants[i];
                    }
                }
                PadInternal(src, res, mode, paramWidths, paramConstants);
            }
            else{
                PadInternal(src, res, mode, paramWidths, null);
            }
            return res;
        }
        /// <summary>
        /// Pad a tensor with given values.
        /// For details, please refer to https://numpy.org/doc/stable/reference/generated/numpy.pad.html?highlight=pad#numpy.pad
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"></param>
        /// <param name="widths"> Number of values padded to the edges of each axis. ((width_1, width_1), … (width_N, width_N)) same pad widths for each axis. </param>
        /// <param name="mode"> The mode of the padding.</param>
        /// <param name="constants"> Used in ‘constant’. The values to set the padded values for each axis. ((value_1, value_1), ... (value_N, value_N)) same pad constants for each axis. Default is 0.</param>
        /// <returns></returns>
        /// <exception cref="InvalidParamException"></exception>
        /// <exception cref="NotImplementedException"></exception>
        public static Tensor<T> Pad<T>(this Tensor<T> src, int width, PadMode mode = PadMode.Constant, 
            double constant = 0) where T : struct, IEquatable<T>, IConvertible
        {
            if(mode == PadMode.Medium || mode == PadMode.Mean){
                throw new NotImplementedException("This padding mode has not been implemented yet. It will be implemented in the future version.");
            }
            int[] paramWidths = new int[src.TLayout.NDim * 2];
            for (int i = 0; i < src.TLayout.NDim; i++){
                paramWidths[i * 2] = paramWidths[i * 2 + 1] = width;
            }
            Tensor<T> res = new Tensor<T>(DeduceLayout(src.TLayout, paramWidths));
            res.TLayout.InitContiguousLayout();
            if (mode == PadMode.Constant)
            {
                double[] paramConstants = new double[paramWidths.Length];
                paramConstants.AsSpan().Fill(constant);
                PadInternal(src, res, mode, paramWidths, paramConstants);
            }
            else{
                PadInternal(src, res, mode, paramWidths, null);
            }
            return res;
        }
        private unsafe static void PadInternal<T>(Tensor<T> src, Tensor<T> dst, PadMode mode, int[] widths, double[]? constants) where T : struct, IEquatable<T>, IConvertible{
            fixed(int* pwidths = widths){
                if(constants is null){
                    PadParam p = new PadParam() { mode = mode, size = widths.Length, widths = new IntPtr(pwidths), constants = IntPtr.Zero };
                    IntPtr status = NativeExecutor.Execute(NativeApi.Pad, src.TMemory, dst.TMemory, src.TLayout, dst.TLayout, new IntPtr(&p), Tensor<T>.Provider);
                    NativeStatus.AssertOK(status);
                }
                else{
                    fixed(double* pconstants = constants){
                        PadParam p = new PadParam() { mode = mode, size = widths.Length, widths = new IntPtr(pwidths), constants = new IntPtr(pconstants) };
                        IntPtr status = NativeExecutor.Execute(NativeApi.Pad, src.TMemory, dst.TMemory, src.TLayout, dst.TLayout, new IntPtr(&p), Tensor<T>.Provider);
                        NativeStatus.AssertOK(status);
                    }
                }
            }
        }
        private static TensorLayout DeduceLayout(TensorLayout src, int[] widths){
            TensorLayout res = new TensorLayout();
            res.DType = src.DType;
            res.NDim = src.NDim;
            for (int i = 0; i < src.NDim; i++) {
                if (widths[i * 2] < 0 || widths[i * 2 + 1] < 0) {
                    throw new InvalidParamException("Padding width must be a positive number");
                }
                res.Shape[i] = src.Shape[i] + widths[i * 2] + widths[i * 2 + 1];
            }
            return res;
        }
    }

    public static partial class Tensor{
        /// <summary>
        /// Pad a tensor with given values.
        /// For details, please refer to https://numpy.org/doc/stable/reference/generated/numpy.pad.html?highlight=pad#numpy.pad
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"> The teneosr to be padded. </param>
        /// <param name="widths"> Number of values padded to the edges of each axis. ((before_1, after_1), … (before_N, after_N)) unique pad widths for each axis. </param>
        /// <param name="mode"> The mode of the padding.</param>
        /// <param name="constants"> Used in ‘constant’. The values to set the padded values for each axis. ((before_1, after_1), ... (before_N, after_N)) unique pad constants for each axis. Default is 0.</param>
        /// <returns></returns>
        /// <exception cref="InvalidParamException"></exception>
        /// <exception cref="NotImplementedException"></exception>
        public static Tensor<T> Pad<T>(Tensor<T> src, (int, int)[] widths, PadMode mode = PadMode.Constant, 
            (double, double)[]? constants = null) where T : struct, IEquatable<T>, IConvertible{
            return src.Pad(widths, mode, constants);
        }
        /// <summary>
        /// Pad a tensor with given values.
        /// For details, please refer to https://numpy.org/doc/stable/reference/generated/numpy.pad.html?highlight=pad#numpy.pad
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"> The teneosr to be padded. </param>
        /// <param name="widths"> Number of values padded to the edges of each axis. ((width_1, width_1), … (width_N, width_N)) same pad widths for each axis. </param>
        /// <param name="mode"> The mode of the padding.</param>
        /// <param name="constants"> Used in ‘constant’. The values to set the padded values for each axis. ((value_1, value_1), ... (value_N, value_N)) same pad constants for each axis. Default is 0.</param>
        /// <returns></returns>
        /// <exception cref="InvalidParamException"></exception>
        /// <exception cref="NotImplementedException"></exception>
        public static Tensor<T> Pad<T>(Tensor<T> src, int[] widths, PadMode mode = PadMode.Constant, 
            double[]? constants = null) where T : struct, IEquatable<T>, IConvertible{
            return src.Pad(widths, mode, constants);
        }
        /// <summary>
        /// Pad a tensor with given values.
        /// For details, please refer to https://numpy.org/doc/stable/reference/generated/numpy.pad.html?highlight=pad#numpy.pad
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="src"> The teneosr to be padded. </param>
        /// <param name="widths"> Number of values padded to the edges of each axis. ((width_1, width_1), … (width_N, width_N)) same pad widths for each axis. </param>
        /// <param name="mode"> The mode of the padding.</param>
        /// <param name="constants"> Used in ‘constant’. The values to set the padded values for each axis. ((value_1, value_1), ... (value_N, value_N)) same pad constants for each axis. Default is 0.</param>
        /// <returns></returns>
        /// <exception cref="InvalidParamException"></exception>
        /// <exception cref="NotImplementedException"></exception>
        public static Tensor<T> Pad<T>(Tensor<T> src, int widths, PadMode mode = PadMode.Constant, 
            double constants = 0) where T : struct, IEquatable<T>, IConvertible{
            return src.Pad(widths, mode, constants);
        }
    }
}