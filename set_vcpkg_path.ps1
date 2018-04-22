Function Get-FileName
{
    [System.Reflection.Assembly]::LoadWithPartialName("System.windows.forms") | Out-Null
    
    $OpenFileDialog = New-Object System.Windows.Forms.OpenFileDialog
    $OpenFileDialog.initialDirectory = $(get-location).Path
    $OpenFileDialog.filter = "CMake (*.cmake)| *.cmake"
	$OpenFileDialog.title = "Select vcpkg.cmake"
    $OpenFileDialog.ShowDialog() | Out-Null
	$OpenFileDialog.filename
}

[System.Windows.Forms.MessageBox]::Show("Please select the vcpkg.cmake file in the next dialog.  
It is usually located in [vcpkg-root]/scripts/buildsystems/","Graphics2",0)

$vcpkgpath = Get-FileName
if($vcpkgpath -eq ""){
	[System.Windows.Forms.MessageBox]::Show("Error setting the path.","Graphics2",0)
	Exit
}

$cmakesettingspath = "./CMakeSettings.json"

$a = Get-Content $cmakesettingspath -raw | ConvertFrom-Json
$a.configurations.variables | % {if($_.name -eq 'CMAKE_TOOLCHAIN_FILE'){$_.value=$vcpkgpath}}
$a | ConvertTo-Json -Depth 20 | set-content $cmakesettingspath

[System.Windows.Forms.MessageBox]::Show("Finished setting vcpkg path in CMakeSettings.json","Graphics2",0)
