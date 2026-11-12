function warnIfFutureIR(modelProto)

%   Copyright 2019-2023 The MathWorks, Inc.

md = nnet.internal.cnn.onnx.NetworkMetadata;
if ~isempty(modelProto.ir_version) && modelProto.ir_version > md.IrVersion
    iWarningWithoutBacktrace('nnet_cnn_onnx:onnx:IrVersion', modelProto.ir_version, md.IrVersion);
end

function iWarningWithoutBacktrace(msgID, varargin)
backtrace = warning('query','backtrace');
warning('off','backtrace');
warning(message(msgID, varargin{:}));
warning(backtrace.state,'backtrace');
end

end