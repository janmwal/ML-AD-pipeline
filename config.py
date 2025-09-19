PREDICTION_THRESHOLDS = {
    'lgbm' : {
        'unthrs' : { # AUC-ROC: 0.890
            'youden' : 0.4537, # Sens=0.746 Spec=0.884 Prec=0.783 F1=0.764
            'sensitivity' : 0.1508, # Sens=0.901 Spec=0.619 Prec=0.569 F1=0.698
            'f1' : 0.4537 # Sens=0.746 Spec=0.884 Prec=0.783 F1=0.764
        },
        'thrs' : { # AUC-ROC: 0.880
            'youden' : 0.3045, # Sens=0.813 Spec=0.781 Prec=0.677 F1=0.739
            'sensitivity' : 0.1535, # Sens=0.902 Spec=0.569 Prec=0.542 F1=0.677
            'f1' : 0.5737 # Sens=0.677 Spec=0.913 Prec=0.815 F1=0.740
        }
    }
}

CLASS_LABELS = {0: "CN", 1: "AD"}
