@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0deploy\openshift\Open-MVP.ps1"
if errorlevel 1 pause
