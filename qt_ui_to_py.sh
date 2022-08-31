# shellcheck disable=SC2006
PYUIC=`which pyuic5`
PYRCC=`which pyrcc5`

MainUI=hf_vision/qt/editor_main_window.ui
DLGAboutUI=hf_vision/qt/dialog_about.ui
RCCUI=hf_vision/qt/regional_detect.qrc

MainPy=hf_vision/qt/editor_main_window.py
DLGAboutPy=hf_vision/qt/dialog_about.py
RCCPy=hf_vision/qt/regional_detect_rc.py

$PYUIC $MainUI -o $MainPy
$PYUIC $DLGAboutUI -o $DLGAboutPy
$PYRCC $RCCUI -o $RCCPy