@echo off

set PYUIC=pyuic5.exe
set PYRCC=pyrcc5.exe

set ALL_UI=editor_main_window dialog_about
set ALL_RC=regional_detect

echo Convert Qt UI...
for %%e in (%ALL_UI%) do (
    echo == %%e.ui ...
    %PYUIC% hf_vision/qt/%%e.ui -o hf_vision/qt/%%e.py
)
echo DONE!

echo.
echo Convert Qt QRC...
for %%e in (%ALL_RC%) do (
    echo == %%e.qrc ...
    %PYRCC% hf_vision/qt/%%e.qrc -o hf_vision/qt/%%e_rc.py
)
echo DONE!

@echo on