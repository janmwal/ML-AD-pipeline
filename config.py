PREDICTION_THRESHOLDS = {
    'lgbm' : {
        'unthrs' : { # AUC-ROC: 0.928
            'youden' : 0.2574, # Sens=0.888 Spec=0.824 Prec=0.738 F1=0.806
            'sensitivity' : 0.2230, # Sens=0.901 Spec=0.805 Prec=0.721 F1=0.801
            'f1' : 0.3558 # Sens=0.819 Spec=0.887 Prec=0.802 F1=0.810
        },
        'thrs' : { # AUC-ROC: 0.865
            'youden' : 0.5099, # Sens=0.706 Spec=0.870 Prec=0.755 F1=0.730
            'sensitivity' : 0.1607, # Sens=0.902 Spec=0.566 Prec=0.541 F1=0.676
            'f1' : 0.5099 # Sens=0.706 Spec=0.870 Prec=0.755 F1=0.730
        }
    }
}

CLASS_LABELS = {0: "CN", 1: "AD"}
