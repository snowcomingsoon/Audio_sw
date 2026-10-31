#define AppName "StudioEQ Master Pro"
#define AppVersion "3.1.0"
#define AppExe "StudioEQ_Master_Pro.exe"
#define PythonHome GetEnv("LOCALAPPDATA") + "\\Programs\\Python\\Python312"
#define FfmpegHome GetEnv("LOCALAPPDATA") + "\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-9.0.2-full_build\\bin"
#define HuggingFaceHub GetEnv("USERPROFILE") + "\\.cache\\huggingface\\hub"

[Setup]
AppId={{B13CCB7D-1C63-4A4B-8D1E-7C402A81C703}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=StudioEQ
DefaultDirName={localappdata}\Programs\StudioEQ Master Pro
DefaultGroupName=StudioEQ Master Pro
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=dist
OutputBaseFilename=StudioEQ_Master_Pro_Setup
SetupIconFile=app_icon.ico
UninstallDisplayIcon={app}\{#AppExe}
ArchitecturesInstallIn64BitMode=x64
ArchitecturesAllowed=x64
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
Uninstallable=yes

[Languages]
Name: "italian"; MessagesFile: "compiler:Languages\Italian.isl"

[Tasks]
Name: "desktopicon"; Description: "Crea un collegamento sul Desktop"; GroupDescription: "Collegamenti aggiuntivi:"; Flags: unchecked

[Files]
Source: "StudioEQ_Master_Pro.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "demucs_pedalboard_master.py"; DestDir: "{app}"; Flags: ignoreversion
Source: "requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "app_icon.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "assets\*"; DestDir: "{app}\assets"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "audio_modules\*"; DestDir: "{app}\audio_modules"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: ".venv\*"; DestDir: "{app}\.venv"; Excludes: "**\tests\**,**\test\**,**\__pycache__\**,*.pyc,*.pyo"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "{#PythonHome}\*"; DestDir: "{app}\python312"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "{#FfmpegHome}\*"; DestDir: "{app}\ffmpeg\bin"; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "{#HuggingFaceHub}\*"; DestDir: "{app}\models\hub"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\StudioEQ Master Pro"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\StudioEQ Master Pro"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Avvia StudioEQ Master Pro"; Flags: postinstall nowait skipifsilent

[Code]
procedure UpdateBundledPythonConfig();
var
  ConfigPath: string;
  Lines: TStringList;
  I: Integer;
begin
  ConfigPath := ExpandConstant('{app}\.venv\pyvenv.cfg');
  if not FileExists(ConfigPath) then
    Exit;

  Lines := TStringList.Create;
  try
    Lines.LoadFromFile(ConfigPath);
    for I := 0 to Lines.Count - 1 do
    begin
      if Pos('home = ', Lines[I]) = 1 then
        Lines[I] := 'home = ' + ExpandConstant('{app}\python312')
      else if Pos('executable = ', Lines[I]) = 1 then
        Lines[I] := 'executable = ' + ExpandConstant('{app}\python312\python.exe')
      else if Pos('command = ', Lines[I]) = 1 then
        Lines[I] := 'command = "' + ExpandConstant('{app}\python312\python.exe') +
          '" -m venv "' + ExpandConstant('{app}\.venv') + '"';
    end;
    Lines.SaveToFile(ConfigPath);
  finally
    Lines.Free;
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    UpdateBundledPythonConfig();
end;
