function [poolSize, stride, padding, paddingValue, dataFormat, numDimsY] = prepareAveragePoolArgs(poolSize_, stride_, padding_, count_include_pad_, numDimsX_)
% Prepares arguments for implementing the ONNX AveragePool operator
%#codegen

% Copyright 2024 The MathWorks, Inc.

    poolSize__              = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(poolSize_);
    stride__                = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(stride_);
    padding__               = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(padding_);
    count_include_pad       = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(count_include_pad_);
    numDimsX                = nnet.internal.cnn.onnx.fcn.coder.ops.extractIfDlarray(numDimsX_);
    
    poolSize    = fliplr(poolSize__(:)');
    stride      = fliplr(stride__(:)');
    
    if isnumeric(padding__)
        % ONNX: [x1_begin, ..., xn_begin, x1_end, ...,xn_end]
        % DLT:  [xn_begin, ..., x1_begin;
        %        xn_end, ..., x1_end]       (Note the fliplr and semicolon)
        padding = fliplr(transpose(reshape(padding__, [], 2)));
    else
        padding = padding__;
    end
    
    if logical(count_include_pad)
        paddingValue = 0;
    else
        paddingValue = 'mean';
    end
    
    dataFormat  = [repmat('S', 1, numDimsX-2) 'CB'];
    numDimsY = numDimsX;
end