% Modification made by:
% Jose Alejandro Ledesma and Stidl Alfonso torres
% Electronic Engineers 
% For graduate work "Evaluation of no-reference quality prediction metrics in videos impaired by authentic distortions"
% Pontificia Universidad Javeriana Cali, Santiago de Cali 2019-2020
% Supervised by:
% Hernán Darío Benítez Restrepo
% Roger Alfonso Gómez Nieto

clc;
close all;
clear all;

% Distribucion de porcentajes para entrenamiento, prueba y numero de iteraciones para buscar los mejores parametros 
Training_Percentage = 0.8;
Test_Percentaje = 0.2;
SVR_Iterations = 350;

% Se cargan los puntajes objetivos 
data_LIVEVQC = load(...
    'D:\JAVERIANA\THESIS\RESULTS\NSTSS\features_vqc.mat');


% Se cargan los puntajes subjetivos 
MOS_LIVEVQC =  load(...
    'D:\JAVERIANA\THESIS\MODIFICACION DE MATRICES\NSTSS\mos_vqc.mat');
data_LIVEVQC=data_LIVEVQC.matriz_videos_features;
MOS_LIVEVQC = MOS_LIVEVQC.mos;

for i=1:100
    tic
% Se genera una matriz con los puntajes de cada video aleatoreamente
    [trainInd,valInd,testInd] = dividerand(161,Training_Percentage,0,Test_Percentaje); 
    %Divido aleatoriamente el 80% de la base de datos     %para entrenamiento
    
    Training_Data = data_LIVEVQC(trainInd,:);
    Training_MOS = MOS_LIVEVQC (trainInd);
    
    Test_Data = data_LIVEVQC(testInd,:);
    Test_MOS = MOS_LIVEVQC (testInd)';
    
% Se entrena el modelo 
    Mdl=fitrsvm(Training_Data,Training_MOS,'Standardize',...
        true,...
        'OptimizeHyperparameters',...
        {'BoxConstraint', 'Epsilon', 'KernelFunction'},...
        'CacheSize','maximal',...
        'HyperparameterOptimizationOptions',struct('UseParallel',1,'MaxObjectiveEvaluations',SVR_Iterations,...
        'ShowPlots',false));
    
% Se evaluan las correlaciones en cada iteracion 
    yfit_LIVE= predict(Mdl,Test_Data);
    R_LIVE = corrcoef(yfit_LIVE,Test_MOS)
    
    %probando con los mismos datos de entrenamiento, el resultado deberia ser cercano a 1
    yfit_SameTraining= predict(Mdl,Training_Data);
    R_SameTraining = corrcoef(yfit_SameTraining,Training_MOS)
    
    Pearson(i)=R_LIVE(1,2)
    Spearman(i)=corr(yfit_LIVE,Test_MOS','Type','Spearman')
    Kendall_COrrelation(i) = corr(yfit_LIVE,Test_MOS','type','Kendall');
    RMSE(i) = sqrt(mean((yfit_LIVE-Test_MOS').^2));
    fprintf('Iteration %d\n',i);
    toc
end
mediana_pearson = median(Pearson);
mediana_spearman = median(Spearman);
mediana_RMSE = median(RMSE);

std_pearson = std(Pearson);
std_Spearman = std(Spearman);
std_RMSE = std(RMSE);

max_pearson = max(Pearson);
max_spearman = max(Spearman);
max_RMSE = min(RMSE);

mean_pearson= mean(Pearson');
mean_Spearman= mean(Spearman');
mean_RMSE= mean(RMSE');

