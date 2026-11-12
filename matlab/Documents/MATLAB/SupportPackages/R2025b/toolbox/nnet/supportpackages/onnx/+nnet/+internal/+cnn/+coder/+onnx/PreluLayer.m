classdef PreluLayer < nnet.layer.Layer
    % Internal, codegen version of nnet.onnx.layer.PreluLayer
    %#codegen
    
    %   Copyright 2017-2024 The MathWorks, Inc.
    properties (Learnable)
        Alpha           % Alpha can be a scalar, or [1 ... 1 C]
    end
    
    properties(Hidden, SetAccess=protected)
        NumChannels
        ChannelDim      
    end
    
    methods
        function this = PreluLayer(name, numChannels, channelDim, initialAlpha)
            this.Name           = name;
            this.NumChannels    = numChannels;
            this.ChannelDim     = channelDim;
            this.Description    = getString(message('nnet_cnn_onnx:onnx:PreluLayerDescription'), matlab.internal.i18n.locale('en_US'));
            this.Type           = getString(message('nnet_cnn_onnx:onnx:PreluLayerType'), matlab.internal.i18n.locale('en_US'));
            if this.NumChannels==1
                % Scalar alpha is used for all channels
                this.Alpha      = single(initialAlpha);                
            else
                shape           = [ones(1,this.ChannelDim-1) this.NumChannels 1];    % Need the trailing 1 in case ChannelDim=1
                this.Alpha     	= zeros(shape, 'single');
                this.Alpha(:)   = initialAlpha;
            end
        end
        
        function Z = predict(layer, X)
            %Typecasting to workaround issue caused by dlnetwork datatype
            Z = max(0,X) + cast(layer.Alpha,'like',X).*min(0,X);
        end        
    end

    methods (Static)
        % This layer does not support online update of learnables
        function tf = matlabCodegenUpdatableLayer(~)
            tf = false;
        end

        function this_cg = matlabCodegenToRedirected(this)
            % Enable objects of class PreluLayer to be passed from MATLAB 
            % to the generated MEX function
            this_cg = nnet.internal.cnn.coder.onnx.PreluLayer(this.Name, this.NumChannels, this.ChannelDim, this.Alpha);
        end
         
        function this = matlabCodegenFromRedirected(this_cg)
            % Enable objects of class PreluLayer to be successfully 
            % returned from the generated MEX function back to MATLAB
            this = nnet.onnx.layer.PreluLayer(this_cg.Name, this_cg.NumChannels, this_cg.ChannelDim, this_cg.Alpha);
        end
     end
end
