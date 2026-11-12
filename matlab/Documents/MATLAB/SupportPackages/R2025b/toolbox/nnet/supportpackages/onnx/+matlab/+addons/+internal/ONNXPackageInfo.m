classdef ONNXPackageInfo < matlab.addons.internal.SupportPackageInfoBase
    % ONNX support package support for MATLAB Compiler.
    
    %   Copyright 2020-2025 The MathWorks, Inc.
    
    methods
        function obj = ONNXPackageInfo()
            obj.baseProduct = 'Deep Learning Toolbox';
            obj.displayName = 'Deep Learning Toolbox Converter for ONNX Model Format';
            obj.name        = 'Deep Learning Toolbox Converter for ONNX Model Format';
            
            % Find the name of the webcammex file based on the platform.
            if ispc
                onnxmexFileName = 'onnxmex.mexw64';
            else
                if ismac
                    macArch = computer('arch');
                    onnxmexFileName = "onnxmex.mex"+macArch;
                else
                    onnxmexFileName = 'onnxmex.mexa64';
                end
            end

            sproot = matlabshared.supportpkg.getSupportPackageRoot();
            % Define all the data that should be deployed from the support
            % package. This includes the actual language data, which will
            % be archived in the CTF.
            obj.mandatoryIncludeList = {...
                fullfile(sproot,'resources','nnet_cnn_onnx'),...
                fullfile(sproot,'bin'),...
                fullfile(sproot,'toolbox','nnet','supportpackages','onnx','+nnet','+internal','+cnn','+onnx',onnxmexFileName),...
                fullfile(sproot,'toolbox','nnet','supportpackages','onnx','+nnet','+internal','+cnn','+onnx','exportONNXNetwork.m'),...
                fullfile(sproot,'toolbox','nnet','supportpackages','onnx','+nnet','+onnx','+layer'),...
                fullfile(sproot,'toolbox','nnet','supportpackages','onnx','+nnet','+internal','+cnn','+onnx','FuncName.m')
                };
         end
    end
end