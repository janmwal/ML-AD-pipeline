PREDICTION_THRESHOLDS = {
    'lgbm' : {
        'unthrs' : { # AUC-ROC: 0.924
            'youden' : 0.4136, # Sens=0.832 Spec=0.880 Prec=0.794 F1=0.813
            'sensitivity' : 0.2703, # Sens=0.901 Spec=0.786 Prec=0.701 F1=0.789
            'f1' : 0.4625 # Sens=0.815 Spec=0.894 Prec=0.811 F1=0.813
        },
        'thrs' : { # AUC-ROC: 0.865
            'youden' : 0.5099, # Sens=0.706 Spec=0.870 Prec=0.755 F1=0.730
            'sensitivity' : 0.1607, # Sens=0.902 Spec=0.566 Prec=0.541 F1=0.676
            'f1' : 0.5099 # Sens=0.706 Spec=0.870 Prec=0.755 F1=0.730
        }
    },
    'extratrees' : {
        'unthrs' : { # AUC-ROC: 0.938
            'youden' : 0.4275, # Sens=0.858 Spec=0.901 Prec=0.829 F1=0.843
            'sensitivity' : 0.3254, # Sens=0.901 Spec=0.853 Prec=0.774 F1=0.833
            'f1' : 0.4275 # Sens=0.858 Spec=0.901 Prec=0.829 F1=0.843
        },
        'thrs' : { # AUC-ROC: 0.870
            'youden' : 0.3695, # Sens=0.779 Spec=0.822 Prec=0.712 F1=0.744
            'sensitivity' : 0.2197, # Sens=0.902 Spec=0.578 Prec=0.548 F1=0.682
            'f1' : 0.3695 # Sens=0.779 Spec=0.822 Prec=0.712 F1=0.744
        }
    }
}

CLASS_LABELS = {0: "CN", 1: "AD"}
