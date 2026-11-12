function warnIfFutureIR(modelProto)

%   Copyright 2019-2025 The MathWorks, Inc.

md = nnet.internal.cnn.onnx.NetworkMetadata;
if ~isempty(modelProto.ir_version) && modelProto.ir_version > md.IrVersion
    nnet.internal.cnn.onnx.util.iWarningWithoutBacktrace('nnet_cnn_onnx:onnx:IrVersion', modelProto.ir_version, md.IrVersion);
end
end