function LayerGraph = importONNXLayers(filename, NameValueArgs)

% Copyright 2018-2024 The MathWorks, Inc.

arguments 
    filename {mustBeTextScalar}
    NameValueArgs.PackageName {mustBeTextScalar} = ''
    NameValueArgs.Namespace {mustBeTextScalar} = ''
    NameValueArgs.OutputLayerType {mustBeTextScalar, mustBeMember(NameValueArgs.OutputLayerType, {'', 'classification', 'regression', 'pixelclassification'})} = ''
    NameValueArgs.ImageInputSize (1, :) {nnet.internal.cnn.onnx.util.validateIONImageInputSize(NameValueArgs.ImageInputSize)} = []
    NameValueArgs.TargetNetwork {mustBeTextScalar, mustBeMember(NameValueArgs.TargetNetwork, {'dlnetwork', 'dagnetwork'})} = 'dagnetwork'
    NameValueArgs.GenerateCustomLayers (1,1) logical = true
    NameValueArgs.InputDataFormats = ''
    NameValueArgs.OutputDataFormats = ''
    NameValueArgs.ImportWeights (1,1) logical {iWarnIfImportWeightsFalse} = true
    NameValueArgs.FoldConstants {mustBeTextScalar, mustBeMember(NameValueArgs.FoldConstants, {'deep', 'shallow', 'none'})} = 'deep'
end 

% Warn about importONNXLayers deprecation
iWarningWithoutBacktrace('nnet_cnn_onnx:onnx:WarnAPIDeprecation', 'importONNXLayers');

% Warn about PackageName argument deprecation
if(~isempty(NameValueArgs.PackageName))
    iWarningWithoutBacktrace('nnet_cnn_onnx:onnx:WarnArgumentDeprecation');
end
% Copy Namespace value to PackageName if Namespace is not empty
if(~isempty(NameValueArgs.Namespace))
    NameValueArgs.PackageName = NameValueArgs.Namespace;
end

nnet.internal.cnn.onnx.setAdditionalResourceLocation(); % For SPKG resource catalog.
[Filename, OutputLayerType, userInputFormats, userOutputFormats, ...
    GenerateCustomLayers, TargetNetwork, UserImageInputSize, ...
    PackageName, OrigPackageName, FoldConstants] = iValidateInputs(filename, NameValueArgs);
% Import
modelProto = nnet.internal.cnn.onnx.ModelProto(Filename);
nnet.internal.cnn.onnx.warnIfFutureIR(modelProto);
iCheckEmptyModel(modelProto,filename);
% Set isForwardONNX to false since for importONNXLayers the expected
% dimension ordering is reverse of ONNX
isInputForwardONNX = false;
CustomLayerContent = "groupedOperators";
modelTranslationIntoLayers = nnet.internal.cnn.onnx.ModelTranslationIntoLayers(modelProto, ...
    OutputLayerType, userInputFormats, userOutputFormats, TargetNetwork, GenerateCustomLayers, ...
    UserImageInputSize, [], PackageName, OrigPackageName, FoldConstants, isInputForwardONNX,...
    CustomLayerContent, false);
LayerGraph = modelTranslationIntoLayers.LayerGraph;
translationIssues = modelTranslationIntoLayers.TranslationIssues;
% Warn if there are unsupported or missing layers
iWarnIfTranslationIssues(translationIssues);
end

function iWarnIfTranslationIssues(translationIssues)
[issueCounts, issueStrings, isPlaceholderIssue] = iParseTranslationIssues(translationIssues);
% Warn if there are unsupported or missing layers
iSummarizeTranslationIssues(issueCounts, issueStrings, isPlaceholderIssue);
end

function [IssueCounts, IssueStrings, placeholderIssueIdx] = iParseTranslationIssues(translationIssues)
% Summarizes the issues encountered during translation, and indicates which
% of the issues are NodeTranslationErrors that resulted in placeholders.
% Report on issues that led to Placeholder creation as well as warnings.
allIssues       = arrayfun(@(x) getMessageString(x), translationIssues, 'UniformOutput', false);
placeholderIssueIdx = arrayfun(@(x) isa(x, 'nnet.internal.cnn.onnx.NodeTranslationError'), translationIssues);
[uniqueIssues, uniqueIdx] = unique(allIssues); 
placeholderIssueIdx = placeholderIssueIdx(uniqueIdx);
IssueStrings    = string(uniqueIssues);

IssueCounts = zeros(numel(IssueStrings), 1);
for i=1:numel(IssueStrings)
    IssueCounts(i) = sum(strcmp(IssueStrings(i), allIssues));
end
end

function iSummarizeTranslationIssues(issueCounts, issueStrings, isPlaceholderIssue)
% Error if any unsupported layers, refer user to importONNXFunction
prologue = "";
epilogue = message('nnet_cnn_onnx:onnx:UseImportONNXFunction').getString();
if numel(issueCounts) > 0
    placeholderIssueStrings = issueStrings(isPlaceholderIssue);
    placeholderIssueCounts = issueCounts(isPlaceholderIssue);
    warningIssueStrings = issueStrings(~isPlaceholderIssue);
    warningIssueCounts = issueCounts(~isPlaceholderIssue);

    placeholderSummary = string.empty;
    if any(isPlaceholderIssue)
        placeholderSummary = iGetIssueSummary(placeholderIssueStrings, placeholderIssueCounts, 'nnet_cnn_onnx:onnx:UnsupportedLayerWarning');
    end

    warningSummary = string.empty;
    if any(~isPlaceholderIssue)
        warningSummary = iGetIssueSummary(warningIssueStrings, warningIssueCounts, 'nnet_cnn_onnx:onnx:TranslationWarningSummary');
    end

    summary = join([placeholderSummary, warningSummary], newline); 
    iWarningWithoutBacktrace('nnet_cnn_onnx:onnx:IssuesDuringImport', prologue, summary, epilogue);
end
end

function issueSummary = iGetIssueSummary(issueStrings, issueCounts, issueID)
    issueSummary = [message(issueID).getString(), newline];
    for i=1:numel(issueCounts)
        issueSummary = join([issueSummary,...
            sprintf("%d operator(s)\t:\t%s", issueCounts(i), issueStrings(i))], newline);
    end
end

function layers = iRemovePlaceholderOutputLayers(placeholderLayers)
pLayerIdx = arrayfun(@(x)~isa(x, 'nnet.onnx.layer.PlaceholderOutputLayer'), placeholderLayers);
layers = placeholderLayers(pLayerIdx);
end

function [Filename, OutputLayerType, userInputFormats, userOutputFormats, ...
    GenerateCustomLayers, TargetNetwork, ImageInputSize, PackageName, OrigPackageName, FoldConstants] = iValidateInputs(filename, NameValueArgs)
Filename                = iValidateFile(filename);
ImageInputSize          = double(NameValueArgs.ImageInputSize(:))';
OutputLayerType         = iValidateOutputLayerType(NameValueArgs.OutputLayerType);
userInputFormats        = nnet.internal.cnn.onnx.util.validateIONDataFormats(NameValueArgs.InputDataFormats, "InputDataFormats");
userOutputFormats       = nnet.internal.cnn.onnx.util.validateIONDataFormats(NameValueArgs.OutputDataFormats, "OutputDataFormats");
[PackageName, OrigPackageName] = iValidatePackageName(NameValueArgs.PackageName, filename);
GenerateCustomLayers    = NameValueArgs.GenerateCustomLayers;
TargetNetwork           = NameValueArgs.TargetNetwork;
FoldConstants           = NameValueArgs.FoldConstants;
end

function iWarnIfImportWeightsFalse(ImportWeights)
if ~ImportWeights
    warning(message("nnet_cnn_onnx:onnx:ImportWeightsIgnored"));
end
end

function [packageName, OrigPackageName] = iValidatePackageName(OrigPackageName, onnxFilename)
if isempty(OrigPackageName)
    [~,OrigPackageName,~] = fileparts(onnxFilename);
end
if isvarname(OrigPackageName)
    packageName = OrigPackageName;
else
    packageName = matlab.lang.makeValidName(OrigPackageName);
end
end

function iCheckEmptyModel(modelProto, filename)
% Check if a layer graph is present in the imported model
if isempty(modelProto.graph)
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:EmptyModel',filename)));
end
end

function OutputLayerType = iValidateOutputLayerType(OutputLayerType)
if ~isempty(OutputLayerType) 
    if isequal(OutputLayerType,'pixelclassification') && ~nnet.internal.cnn.onnx.isInstalledCVST
        throwAsCaller(MException(message('nnet_cnn_onnx:onnx:noCVSTForPixelClassification')));
    end
end
end

function Filepath = iValidateFile(Filename)
if ~(isa(Filename,'char') || isa(Filename,'string'))
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:FirstArgString')));
end
% Check if the file extension is valid (i.e., if the filename ends with '.onnx')
[~,~,ext] = fileparts(Filename);
if (~strcmp(char(ext),'.onnx'))
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:InvalidFileExtension', Filename)));
end
Filepath = which(char(Filename));
if isempty(Filepath) && exist(Filename, 'file')
    Filepath = char(Filename);
end
if ~exist(Filepath, 'file')
    throwAsCaller(MException(message('nnet_cnn_onnx:onnx:FileNotFound', Filename)));
end
end

function iWarningWithoutBacktrace(msgID, varargin)
backtrace = warning('query','backtrace');
warning('off','backtrace');
warning(message(msgID, varargin{:}));
warning(backtrace.state,'backtrace');
end
