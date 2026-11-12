classdef TranslatorForAbs < nnet.internal.cnn.onnx.TranslatorForUnsupportedSISOPassthroughOp
    
    % Copyright 2021 The MathWorks, Inc.
    methods
        % Do initial setup. this.Node has been set already.
        function initialize(this)
            this.CanPropagateSingletonFormats = true;
        end
    end
end
