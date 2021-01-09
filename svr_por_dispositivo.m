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
SVR_Iterations = 350;

% Se cargan los puntajes objetivos 
data_LIVEVQC = load(...
    'D:\JAVERIANA\THESIS\MODIFICACION DE MATRICES\BRISQUE\features_qualcomm_dispositivo_arreglado.mat');

% Se cargan los puntajes subjetivos 
MOS_LIVEVQC =  load(...
    'D:\JAVERIANA\THESIS\MODIFICACION DE MATRICES\BRISQUE\mos_qualcomm_por_dispositivos_modificado.mat');
data_LIVEVQC=data_LIVEVQC.features_qualcomm;
MOS_LIVEVQC = MOS_LIVEVQC.mos_dispositivos_modificado;

% Se divide la infromacion para cada uno de los dispositivos 
gs5=load('D:\JAVERIANA\THESIS\RESULTS\BRISQUE\Qualcomm\SVR POR PHONES\gs5.mat');
gs5_f=gs5.gs5_features;
gs5_m=-gs5.gs5_mos+100;

gs6=load('D:\JAVERIANA\THESIS\RESULTS\BRISQUE\Qualcomm\SVR POR PHONES\gs6.mat');
gs6_f=gs6.gs6_features;
gs6_m=-gs6.gs6_mos+100;

htc=load('D:\JAVERIANA\THESIS\RESULTS\BRISQUE\Qualcomm\SVR POR PHONES\htc.mat');
htc_f=htc.htc_features;
htc_m=-htc.htc_mos+100;

iphone=load('D:\JAVERIANA\THESIS\RESULTS\BRISQUE\Qualcomm\SVR POR PHONES\iphone.mat');
iphone_f=iphone.iphone_features;
iphone_m=-iphone.iphone_mos+100;

lgg2=load('D:\JAVERIANA\THESIS\RESULTS\BRISQUE\Qualcomm\SVR POR PHONES\lgg2.mat');
lgg2_f=lgg2.lgg2_features;
lgg2_m=-lgg2.lgg2_mos+100;

lumia=load('D:\JAVERIANA\THESIS\RESULTS\BRISQUE\Qualcomm\SVR POR PHONES\lumia.mat');
lumia_f=lumia.lumia_features;
lumia_m=-lumia.lumia_mos+100;

note=load('D:\JAVERIANA\THESIS\RESULTS\BRISQUE\Qualcomm\SVR POR PHONES\note.mat');
note_f=note.note_features;
note_m=-note.note_mos+100;

oppo=load('D:\JAVERIANA\THESIS\RESULTS\BRISQUE\Qualcomm\SVR POR PHONES\oppo.mat');
oppo_f=oppo.oppo_features;
oppo_m=-oppo.oppo_mos+100;

for i=1:100
    tic

   % Se generan matrices con los puntajes de cada escena aleatoreamente
    [trainInd,valInd,testInd] = dividerand(54,Training_Percentage,0,Test_Percentaje);     
    Training_Data = data_LIVEVQC(trainInd,:);
    Training_MOS = MOS_LIVEVQC (trainInd);
    
    [trainInd,valInd,testInd] = dividerand(54,Training_Percentage,0,Test_Percentaje); 
    Test_Data = data_LIVEVQC(testInd,:);
    Test_MOS = MOS_LIVEVQC (testInd)';
    
    [trainInd,valInd,testInd] = dividerand(22,Training_Percentage,0,Test_Percentaje); 
    Test_Data_gs5 = gs5_f(testInd,:);
    Test_MOS_gs5  = gs5_m (testInd)';

    [trainInd,valInd,testInd] = dividerand(33,Training_Percentage,0,Test_Percentaje); 
    Test_Data_gs6 = gs6_f(testInd,:);
    Test_MOS_gs6  = gs6_m (testInd)';

    [trainInd,valInd,testInd] = dividerand(31,Training_Percentage,0,Test_Percentaje); 
    Test_Data_htc= htc_f(testInd,:);
    Test_MOS_htc= htc_m (testInd)';
    
    [trainInd,valInd,testInd] = dividerand(21,Training_Percentage,0,Test_Percentaje); 
    Test_Data_iphone= iphone_f(testInd,:);
    Test_MOS_iphone= iphone_m (testInd)';
    
    [trainInd,valInd,testInd] = dividerand(31,Training_Percentage,0,Test_Percentaje); 
    Test_Data_lgg2= lgg2_f(testInd,:);
    Test_MOS_lgg2= lgg2_m (testInd)';
    
    [trainInd,valInd,testInd] = dividerand(16,Training_Percentage,0,Test_Percentaje); 
    Test_Data_lumia= lumia_f(testInd,:);
    Test_MOS_lumia= lumia_m (testInd)';
    
    [trainInd,valInd,testInd] = dividerand(22,Training_Percentage,0,Test_Percentaje); 
    Test_Data_note= note_f(testInd,:);
    Test_MOS_note= note_m (testInd)';
    
    [trainInd,valInd,testInd] = dividerand(31,Training_Percentage,0,Test_Percentaje); 
    Test_Data_oppo= oppo_f(testInd,:);
    Test_MOS_oppo= oppo_m (testInd)';
    
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
    R_LIVE = corrcoef(yfit_LIVE,Test_MOS);
    
    yfit_LIVE_gs5= predict(Mdl,Test_Data_gs5);
    R_LIVE_gs5 = corrcoef(yfit_LIVE_gs5,Test_MOS_gs5);
    
    yfit_LIVE_gs6= predict(Mdl,Test_Data_gs6);
    R_LIVE_gs6 = corrcoef(yfit_LIVE_gs6,Test_MOS_gs6);
    
    yfit_LIVE_htc= predict(Mdl,Test_Data_htc);
    R_LIVE_htc = corrcoef(yfit_LIVE_htc,Test_MOS_htc);
    
    yfit_LIVE_iphone= predict(Mdl,Test_Data_iphone);
    R_LIVE_iphone= corrcoef(yfit_LIVE_iphone,Test_MOS_iphone);
    
    yfit_LIVE_lgg2= predict(Mdl,Test_Data_lgg2);
    R_LIVE_lgg2 = corrcoef(yfit_LIVE_lgg2,Test_MOS_lgg2);
    
    yfit_LIVE_lumia= predict(Mdl,Test_Data_lumia);
    R_LIVE_lumia = corrcoef(yfit_LIVE_lumia,Test_MOS_lumia);
    
    yfit_LIVE_note= predict(Mdl,Test_Data_note);
    R_LIVE_note = corrcoef(yfit_LIVE_note,Test_MOS_note);
    
    yfit_LIVE_oppo= predict(Mdl,Test_Data_oppo);
    R_LIVE_oppo = corrcoef(yfit_LIVE_oppo,Test_MOS_oppo);
    
    %probando con los mismos datos de entrenamiento, el resultado deberia ser cercano a 1
    yfit_SameTraining= predict(Mdl,Training_Data);
    R_SameTraining = corrcoef(yfit_SameTraining,Training_MOS);
    
    Pearson_all(i)=R_LIVE(1,2);
    Spearman_all(i)=corr(yfit_LIVE,Test_MOS','Type','Spearman');
    RMSE_all(i) = sqrt(mean((yfit_LIVE-Test_MOS').^2));
    
    Pearson_gs5(i)=R_LIVE_gs5(1,2);
    Spearman_gs5(i)=corr(yfit_LIVE_gs5,Test_MOS_gs5','Type','Spearman');
    RMSE_gs5(i) = sqrt(mean((yfit_LIVE_gs5-Test_MOS_gs5').^2));
    
    Pearson_gs6(i)=R_LIVE_gs6(1,2);
    Spearman_gs6(i)=corr(yfit_LIVE_gs6,Test_MOS_gs6','Type','Spearman');
    RMSE_gs6(i) = sqrt(mean((yfit_LIVE_gs6-Test_MOS_gs6').^2));
    
    Pearson_htc(i)=R_LIVE_htc(1,2);
    Spearman_htc(i)=corr(yfit_LIVE_htc,Test_MOS_htc','Type','Spearman');
    RMSE_htc(i) = sqrt(mean((yfit_LIVE_htc-Test_MOS_htc').^2));
    
    Pearson_iphone(i)=R_LIVE_iphone(1,2);
    Spearman_iphone(i)=corr(yfit_LIVE_iphone,Test_MOS_iphone','Type','Spearman');
    RMSE_iphone(i) = sqrt(mean((yfit_LIVE_iphone-Test_MOS_iphone').^2));
    
    Pearson_lgg2(i)=R_LIVE_lgg2(1,2);
    Spearman_lgg2(i)=corr(yfit_LIVE_lgg2,Test_MOS_lgg2','Type','Spearman');
    RMSE_lgg2(i) = sqrt(mean((yfit_LIVE_lgg2-Test_MOS_lgg2').^2));
    
    Pearson_lumia(i)=R_LIVE_lumia(1,2);
    Spearman_lumia(i)=corr(yfit_LIVE_lumia,Test_MOS_lumia','Type','Spearman');
    RMSE_lumia(i) = sqrt(mean((yfit_LIVE_lumia-Test_MOS_lumia').^2));
    
    Pearson_note(i)=R_LIVE_note(1,2);
    Spearman_note(i)=corr(yfit_LIVE_note,Test_MOS_note','Type','Spearman');
    RMSE_note(i) = sqrt(mean((yfit_LIVE_note-Test_MOS_note').^2));
    
    Pearson_oppo(i)=R_LIVE_oppo(1,2);
    Spearman_oppo(i)=corr(yfit_LIVE_oppo,Test_MOS_oppo','Type','Spearman');
    RMSE_oppo(i) = sqrt(mean((yfit_LIVE_oppo-Test_MOS_oppo').^2));    
    
    
    fprintf('Iteration %d\n',i);

end

mediana_Pearson_all= median(Pearson_all); mediana_Spearman_all= median(Spearman_all); mediana_RMSE_all= median(RMSE_all);
std_Pearson_all= std(Pearson_all); std_Spearman_all= std(Spearman_all); std_RMSE_all= std(RMSE_all);
T_all = table(mediana_Pearson_all,std_Pearson_all,mediana_Spearman_all,std_Spearman_all,mediana_RMSE_all,std_RMSE_all)

mediana_Pearson_gs5= median(Pearson_gs5); mediana_Spearman_gs5= median(Spearman_gs5); mediana_RMSE_gs5= median(RMSE_gs5);
std_Pearson_gs5= std(Pearson_gs5); std_Spearman_gs5= std(Spearman_gs5); std_RMSE_gs5= std(RMSE_gs5);
T_gs5= table(mediana_Pearson_gs5,std_Pearson_gs5,mediana_Spearman_gs5,std_Spearman_gs5,mediana_RMSE_gs5,std_RMSE_gs5)

mediana_Pearson_gs6= median(Pearson_gs6); mediana_Spearman_gs6= median(Spearman_gs6); mediana_RMSE_gs6= median(RMSE_gs6);
std_Pearson_gs6= std(Pearson_gs6); std_Spearman_gs6= std(Spearman_gs6); std_RMSE_gs6= std(RMSE_gs6);
T_gs6= table(mediana_Pearson_gs6,std_Pearson_gs6,mediana_Spearman_gs6,std_Spearman_gs6,mediana_RMSE_gs6,std_RMSE_gs6)

mediana_Pearson_htc= median(Pearson_htc); mediana_Spearman_htc= median(Spearman_htc); mediana_RMSE_htc= median(RMSE_htc);
std_Pearson_htc= std(Pearson_htc); std_Spearman_htc= std(Spearman_htc); std_RMSE_htc= std(RMSE_htc);
T_htc= table(mediana_Pearson_htc,std_Pearson_htc,mediana_Spearman_htc,std_Spearman_htc,mediana_RMSE_htc,std_RMSE_htc)

mediana_Pearson_iphone= median(Pearson_iphone); mediana_Spearman_iphone= median(Spearman_iphone); mediana_RMSE_iphone= median(RMSE_iphone);
std_Pearson_iphone= std(Pearson_iphone); std_Spearman_iphone= std(Spearman_iphone); std_RMSE_iphone= std(RMSE_iphone);
T_iphone= table(mediana_Pearson_iphone,std_Pearson_iphone,mediana_Spearman_iphone,std_Spearman_iphone,mediana_RMSE_iphone,std_RMSE_iphone)

mediana_Pearson_lgg2= median(Pearson_lgg2); mediana_Spearman_lgg2= median(Spearman_lgg2); mediana_RMSE_lgg2= median(RMSE_lgg2);
std_Pearson_lgg2= std(Pearson_lgg2); std_Spearman_lgg2= std(Spearman_lgg2); std_RMSE_lgg2= std(RMSE_lgg2);
T_lgg2= table(mediana_Pearson_lgg2,std_Pearson_lgg2,mediana_Spearman_lgg2,std_Spearman_lgg2,mediana_RMSE_lgg2,std_RMSE_lgg2)

mediana_Pearson_lumia= median(Pearson_lumia); mediana_Spearman_lumia= median(Spearman_lumia); mediana_RMSE_lumia= median(RMSE_lumia);
std_Pearson_lumia= std(Pearson_lumia); std_Spearman_lumia= std(Spearman_lumia); std_RMSE_lumia= std(RMSE_lumia);
T_lumia= table(mediana_Pearson_lumia,std_Pearson_lumia,mediana_Spearman_lumia,std_Spearman_lumia,mediana_RMSE_lumia,std_RMSE_lumia)

mediana_Pearson_note= median(Pearson_note); mediana_Spearman_note= median(Spearman_note); mediana_RMSE_note= median(RMSE_note);
std_Pearson_note= std(Pearson_note); std_Spearman_note= std(Spearman_note); std_RMSE_note= std(RMSE_note);
T_note= table(mediana_Pearson_note,std_Pearson_note,mediana_Spearman_note,std_Spearman_note,mediana_RMSE_note,std_RMSE_note)

mediana_Pearson_oppo= median(Pearson_oppo); mediana_Spearman_oppo= median(Spearman_oppo); mediana_RMSE_oppo= median(RMSE_oppo);
std_Pearson_oppo= std(Pearson_oppo); std_Spearman_oppo= std(Spearman_oppo); std_RMSE_oppo= std(RMSE_oppo);
T_oppo= table(mediana_Pearson_oppo,std_Pearson_oppo,mediana_Spearman_oppo,std_Spearman_oppo,mediana_RMSE_oppo,std_RMSE_oppo)

toc


