param([string]$OutputPath)
# Native screenshot fallback: the bundled Computer Use runtimes failed to initialize.
# Capture only the real Nsight Compute window, never synthesize its contents.
Add-Type -AssemblyName System.Drawing
Add-Type @'
using System;
using System.Runtime.InteropServices;
public class CaptureWindow {
 [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left,Top,Right,Bottom; }
 [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd,out RECT rect);
 [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr hWnd,IntPtr hdc,uint flags);
 [DllImport("user32.dll")] public static extern bool SetProcessDpiAwarenessContext(IntPtr context);
 [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hWnd);
 [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd,int mode);
}
'@
$targetProcess = Get-Process -Name 'ncu-ui' | Where-Object {$_.MainWindowHandle -ne 0} | Select-Object -First 1
if (-not $targetProcess) { throw 'No Nsight Compute window' }
[void][CaptureWindow]::SetProcessDpiAwarenessContext([IntPtr](-4))
[void][CaptureWindow]::ShowWindow($targetProcess.MainWindowHandle,3)
[void][CaptureWindow]::SetForegroundWindow($targetProcess.MainWindowHandle)
Start-Sleep -Milliseconds 700
$rect = New-Object CaptureWindow+RECT
[void][CaptureWindow]::GetWindowRect($targetProcess.MainWindowHandle,[ref]$rect)
$bitmap = New-Object System.Drawing.Bitmap(($rect.Right-$rect.Left),($rect.Bottom-$rect.Top))
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$graphics.CopyFromScreen($rect.Left,$rect.Top,0,0,$bitmap.Size)
$bitmap.Save($OutputPath,[System.Drawing.Imaging.ImageFormat]::Png)
$graphics.Dispose()
$bitmap.Dispose()
Write-Output $OutputPath
