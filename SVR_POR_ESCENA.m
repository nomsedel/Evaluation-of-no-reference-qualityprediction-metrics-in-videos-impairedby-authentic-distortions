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


tic

% Distribucion de porcentajes para entrenamiento, prueba y numero de iteraciones para buscar los mejores parametros 
Training_Percentage = 0.8;
Test_Percentaje = 0.2;
SVR_Iterations = 1;


% Se cargan los puntajes objetivos 
data_LIVEVQC = load(...
    'D:\JAVERIANA\THESIS\MODIFICACION DE MATRICES\TLVQM\cvd_escena\features.mat');
% Se cargan los puntajes subjetivos 
MOS_LIVEVQC =  load(...
    'D:\JAVERIANA\THESIS\MODIFICACION DE MATRICES\TLVQM\cvd_escena\mos.mat');

% Se divide la infromacion para cada una de las escenas
data_LIVEVQC_city=data_LIVEVQC.features_city;
data_LIVEVQC_talking=data_LIVEVQC.features_talking;
data_LIVEVQC_newspaper=data_LIVEVQC.features_newspaper;
data_LIVEVQC_traffic=data_LIVEVQC.features_traffic;
data_LIVEVQC_television=data_LIVEVQC.features_television;
MOS_LIVEVQC_city = MOS_LIVEVQC.mos_city;
MOS_LIVEVQC_talking= MOS_LIVEVQC.mos_talking;
MOS_LIVEVQC_newspaper= MOS_LIVEVQC.mos_newspaper;
MOS_LIVEVQC_traffic= MOS_LIVEVQC.mos_traffic;
MOS_LIVEVQC_television= MOS_LIVEVQC.mos_television;

for i=1:1
    i
    
    % Se generan matrices con los puntajes de cada escena aleatoreamente
    [trainInd,valInd,testInd] = dividerand(77,Training_Percentage,0,Test_Percentaje);     
    Training_DataC1 = data_LIVEVQC_city(trainInd,:);
    Training_MOSC1 = MOS_LIVEVQC_city (trainInd);
    Test_DataC1 = data_LIVEVQC_city(testInd,:);
    Test_MOSC1 = MOS_LIVEVQC_city (testInd);
    
    [trainInd,valInd,testInd] = dividerand(77,Training_Percentage,0,Test_Percentaje);      
    Training_DataC2 = data_LIVEVQC_talking(trainInd,:);
    Training_MOSC2 = MOS_LIVEVQC_talking (trainInd);
    Test_DataC2 = data_LIVEVQC_talking(testInd,:);
    Test_MOSC2 = MOS_LIVEVQC_talking (testInd);
    
    [trainInd,valInd,testInd] = dividerand(34,Training_Percentage,0,Test_Percentaje);  
    Training_DataC3 = data_LIVEVQC_newspaper(trainInd,:);
    Training_MOSC3 = MOS_LIVEVQC_newspaper (trainInd);
    Test_DataC3 = data_LIVEVQC_newspaper(testInd,:);
    Test_MOSC3 = MOS_LIVEVQC_newspaper (testInd);
    
    [trainInd,valInd,testInd] = dividerand(34,Training_Percentage,0,Test_Percentaje);    
    Training_DataC4 = matriz_4_Focus(trainInd,:);
    Training_MOSC4 = mos_Focus (trainInd);
    Test_DataC4 = matriz_4_Focus(testInd,:);
    Test_MOSC4 = mos_Focus (testInd);
    
    [trainInd,valInd,testInd] = dividerand(34,Training_Percentage,0,Test_Percentaje);     
    Training_DataC5 = matriz_5_Sharpness(trainInd,:);
    Training_MOSC5 = mos_Sharpness (trainInd);
    Test_DataC5 = matriz_5_Sharpness(testInd,:);
    Test_MOSC5 = mos_Sharpness (testInd);
    
    [trainInd,valInd,testInd] = dividerand(34,Training_Percentage,0,Test_Percentaje);       
    Training_DataC6 = matriz_6_Stabilization(trainInd,:);
    Training_MOSC6 = mos_Stabilization (trainInd);
    Test_DataC6 = matriz_6_Stabilization(testInd,:);
    Test_MOSC6 = mos_Stabilization (testInd);

    % Se concatena toda la informacion para un analisis general 
    Training_Data=[Training_DataC1;Training_DataC2;Training_DataC3;Training_DataC4;Training_DataC5;Training_DataC6];
    Training_MOS=[Training_MOSC1;Training_MOSC2;Training_MOSC3;Training_MOSC4;Training_MOSC5;Training_MOSC6];
 
    Test_Data=[Test_DataC1;Test_DataC2;Test_DataC3;Test_DataC4;Test_DataC5;Test_DataC6];
    Test_MOS=[Test_MOSC1;Test_MOSC2;Test_MOSC3;Test_MOSC4;Test_MOSC5;Test_MOSC6];
    
    % Se entrena el modelo 
    Mdl=fitrsvm(Training_Data,Training_MOS,'Standardize',...
        false,...
        'OptimizeHyperparameters',...
        {'BoxConstraint', 'Epsilon', 'KernelFunction'},...
        'CacheSize','maximal',...
        'HyperparameterOptimizationOptions',struct('UseParallel',1,'MaxObjectiveEvaluations',SVR_Iterations,...
        'ShowPlots',false));
    
% Se evaluan las correlaciones en cada iteracion 
    yfit_LIVE_all= predict(Mdl,Test_Data);
    R_LIVE_all = corrcoef(yfit_LIVE_all,Test_MOS');
    
    yfit_LIVE_artefacts= predict(Mdl,Test_DataC1);
    R_LIVE_artefacts = corrcoef(yfit_LIVE_artefacts,Test_MOSC1');
    
    yfit_LIVE_color= predict(Mdl,Test_DataC2);
    R_LIVE_color = corrcoef(yfit_LIVE_color,Test_MOSC2');
    
    yfit_LIVE_exposure= predict(Mdl,Test_DataC3);
    R_LIVE_exposure = corrcoef(yfit_LIVE_exposure,Test_MOSC3');
    
    yfit_LIVE_focus= predict(Mdl,Test_DataC4);
    R_LIVE_focus= corrcoef(yfit_LIVE_focus,Test_MOSC4');
    
    yfit_LIVE_sharpness= predict(Mdl,Test_DataC5);
    R_LIVE_sharpness= corrcoef(yfit_LIVE_sharpness,Test_MOSC5');
    
    yfit_LIVE_stabilization= predict(Mdl,Test_DataC6);
    R_LIVE_stabilization= corrcoef(yfit_LIVE_stabilization,Test_MOSC6');
    
    
        %probando con los mismos datos de entrenamiento, el resultado deberia ser cercano a 1
    yfit_SameTraining= predict(Mdl,Training_Data);
    R_SameTraining = corrcoef(yfit_SameTraining,Training_MOS);
    
    Pearson_all(i)=R_LIVE_all(1,2);
    Spearman_all(i)=corr(yfit_LIVE_all,Test_MOS,'Type','Spearman');
    Kendall_COrrelation_all(i) = corr(yfit_LIVE_all,Test_MOS,'type','Kendall');
    RMSE_all(i) = sqrt(mean((yfit_LIVE_all-Test_MOS).^2));
    
    Pearson_artefacts(i)=R_LIVE_artefacts(1,2);
    Spearman_artefacts(i)=corr(yfit_LIVE_artefacts,Test_MOSC1,'Type','Spearman');
    Kendall_COrrelation_artefacts(i) = corr(yfit_LIVE_artefacts,Test_MOSC1,'type','Kendall');
    RMSE_artefacts(i) = sqrt(mean((yfit_LIVE_artefacts-Test_MOSC1).^2));
    
    Pearson_color(i)=R_LIVE_color(1,2);
    Spearman_color(i)=corr(yfit_LIVE_color,Test_MOSC2,'Type','Spearman');
    Kendall_COrrelation_color(i) = corr(yfit_LIVE_color,Test_MOSC2,'type','Kendall');
    RMSE_color(i) = sqrt(mean((yfit_LIVE_color-Test_MOSC2).^2));
    
    Pearson_exposure (i)=R_LIVE_exposure(1,2);
    Spearman_exposure (i)=corr(yfit_LIVE_exposure,Test_MOSC3,'Type','Spearman');
    Kendall_COrrelation_exposure (i) = corr(yfit_LIVE_exposure,Test_MOSC3,'type','Kendall');
    RMSE_exposure (i) = sqrt(mean((yfit_LIVE_exposure-Test_MOSC3).^2));
    
    Pearson_focus(i)=R_LIVE_focus(1,2);
    Spearman_focus(i)=corr(yfit_LIVE_focus,Test_MOSC4,'Type','Spearman');
    Kendall_COrrelation_focus(i) = corr(yfit_LIVE_focus,Test_MOSC4,'type','Kendall');
    RMSE_focus(i) = sqrt(mean((yfit_LIVE_focus-Test_MOSC4).^2));
    
    Pearson_sharpness(i)=R_LIVE_sharpness(1,2);
    Spearman_sharpness(i)=corr(yfit_LIVE_sharpness,Test_MOSC5,'Type','Spearman');
    Kendall_COrrelation_sharpness(i) = corr(yfit_LIVE_sharpness,Test_MOSC5,'type','Kendall');
    RMSE_sharpness(i) = sqrt(mean((yfit_LIVE_sharpness-Test_MOSC5).^2));
    
    Pearson_stabilization(i)=R_LIVE_stabilization(1,2);
    Spearman_stabilization(i)=corr(yfit_LIVE_stabilization,Test_MOSC6,'Type','Spearman');
    Kendall_COrrelation_stabilization(i) = corr(yfit_LIVE_stabilization,Test_MOSC6,'type','Kendall');
    RMSE_stabilization(i) = sqrt(mean((yfit_LIVE_stabilization-Test_MOSC6).^2));    
    
    fprintf('Iteration %d\n',i);
    
end
mediana_Pearson_all= median(Pearson_all); mediana_Spearman_all= median(Spearman_all); mediana_RMSE_all= median(RMSE_all);
std_Pearson_all= std(Pearson_all); std_Spearman_all= std(Spearman_all); std_RMSE_all= std(RMSE_all);
T_all = table(mediana_Pearson_all,std_Pearson_all,mediana_Spearman_all,std_Spearman_all,mediana_RMSE_all,std_RMSE_all)


mediana_Pearson_artefacts= median(Pearson_artefacts); mediana_Spearman_artefacts= median(Spearman_artefacts); mediana_RMSE_artefacts= median(RMSE_artefacts);
std_Pearson_artefacts= std(Pearson_artefacts); std_Spearman_artefacts= std(Spearman_artefacts); std_RMSE_artefacts= std(RMSE_artefacts);
T_artefacts = table(mediana_Pearson_artefacts,std_Pearson_artefacts,mediana_Spearman_artefacts,std_Spearman_artefacts,mediana_RMSE_artefacts,std_RMSE_artefacts)


mediana_Pearson_color= median(Pearson_color); mediana_Spearman_color= median(Spearman_color); mediana_RMSE_color= median(RMSE_color);
std_Pearson_color= std(Pearson_color); std_Spearman_color= std(Spearman_color); std_RMSE_color= std(RMSE_color);
T_color= table(mediana_Pearson_color,std_Pearson_color,mediana_Spearman_color,std_Spearman_color,mediana_RMSE_color,std_RMSE_color)


mediana_Pearson_exposure= median(Pearson_exposure); mediana_Spearman_exposure= median(Spearman_exposure); mediana_RMSE_exposure= median(RMSE_exposure);
std_Pearson_exposure= std(Pearson_exposure); std_Spearman_exposure= std(Spearman_exposure); std_RMSE_exposure= std(RMSE_exposure);
T_exposure= table(mediana_Pearson_exposure,std_Pearson_exposure,mediana_Spearman_exposure,std_Spearman_exposure,mediana_RMSE_exposure,std_RMSE_exposure)


mediana_Pearson_focus= median(Pearson_focus); mediana_Spearman_focus= median(Spearman_focus); mediana_RMSE_focus= median(RMSE_focus);
std_Pearson_focus= std(Pearson_focus); std_Spearman_focus= std(Spearman_focus); std_RMSE_focus= std(RMSE_focus);
T_focus= table(mediana_Pearson_focus,std_Pearson_focus,mediana_Spearman_focus,std_Spearman_focus,mediana_RMSE_focus,std_RMSE_focus)


mediana_Pearson_sharpness= median(Pearson_sharpness); mediana_Spearman_sharpness= median(Spearman_sharpness); mediana_RMSE_sharpness= median(RMSE_sharpness);
std_Pearson_sharpness= std(Pearson_sharpness); std_Spearman_sharpness= std(Spearman_sharpness); std_RMSE_sharpness= std(RMSE_sharpness);
T_sharpness= table(mediana_Pearson_sharpness,std_Pearson_sharpness,mediana_Spearman_sharpness,std_Spearman_sharpness,mediana_RMSE_sharpness,std_RMSE_sharpness)


mediana_Pearson_stabilization= median(Pearson_stabilization); mediana_Spearman_stabilization= median(Spearman_stabilization); mediana_RMSE_stabilization= median(RMSE_stabilization);
std_Pearson_stabilization= std(Pearson_stabilization); std_Spearman_stabilization= std(Spearman_stabilization); std_RMSE_stabilization= std(RMSE_stabilization);
T_stabilization= table(mediana_Pearson_stabilization,std_Pearson_stabilization,mediana_Spearman_stabilization,std_Spearman_stabilization,mediana_RMSE_stabilization,std_RMSE_stabilization)

toc
