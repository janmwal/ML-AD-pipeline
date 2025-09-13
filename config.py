PREDICTION_THRESHOLDS = {
    'lgbm' : {
        'unthrs' : { # AUC-ROC: 0.906
            'youden' : 0.3154, # Sens=0.828 Spec=0.841 Prec=0.744 F1=0.784
            'sensitivity' : 0.1767, # Sens=0.901 Spec=0.680 Prec=0.611 F1=0.728
            'f1' : 0.4268 # Sens=0.763 Spec=0.901 Prec=0.812 F1=0.787
        },
        'thrs' : { # AUC-ROC: 0.877
            'youden' : 0.3416, # Sens=0.817 Spec=0.786 Prec=0.683 F1=0.744
            'sensitivity' : 0.1776, # Sens=0.902 Spec=0.641 Prec=0.587 F1=0.711
            'f1' : 0.3622 # Sens=0.800 Spec=0.800 Prec=0.694 F1=0.743
        }
    },
    'extratrees' : {
        'unthrs' : { # AUC-ROC: 0.918
            'youden' : 0.3714, # Sens=0.862 Spec=0.836 Prec=0.746 F1=0.800
            'sensitivity' : 0.3046, # Sens=0.901 Spec=0.749 Prec=0.668 F1=0.767
            'f1' : 0.3834 # Sens=0.845 Spec=0.848 Prec=0.757 F1=0.798
        },
        'thrs' : { # AUC-ROC: 0.872
            'youden' : 0.4085, # Sens=0.740 Spec=0.841 Prec=0.725 F1=0.733
            'sensitivity' : 0.2017, # Sens=0.902 Spec=0.571 Prec=0.544 F1=0.678
            'f1' : 0.4040 # Sens=0.740 Spec=0.839 Prec=0.722 F1=0.731
        }
    }

}
CLASS_LABELS = {0: "CN", 1: "AD"}

