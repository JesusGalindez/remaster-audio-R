on adding folder items to this_folder after receiving added_items
	repeat with the_item in added_items
		try
			set the_path to POSIX path of the_item
			
			# Process only video files that are not already remastered
			if (the_path ends with ".mov" or the_path ends with ".mp4" or the_path ends with ".m4v") and not (the_path contains "_remastered") then
				
				# Wait for QuickTime to finish writing the file to disk
				set last_size to -1
				set current_size to 0
				repeat while current_size is not equal to last_size
					set last_size to current_size
					delay 3
					set current_size to (size of (info for the_item))
				end repeat
				
				# Build output path (replace extension with _remastered.mp4)
				set output_path to text 1 thru -5 of the_path & "_remastered.mp4"
				
				# Run python remasterer with AI Lip-Sync enabled
				do shell script "python3 '/Users/albertogalindez/.gemini/Antigravity/Remaster Audio/remasterer.py' process " & quoted form of the_path & " " & quoted form of output_path & " --auto-sync-lips"
				
			end if
		on error errStr number errorNumber
			do shell script "echo " & quoted form of (errStr & " (Error: " & errorNumber & ")") & " >> '/Users/albertogalindez/.gemini/Antigravity/Remaster Audio/folder_action_error.log'"
		end try
	end repeat
end adding folder items to
