# Security Policy

## Supported Versions

Currently supported versions of STCM Mapper with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Security Considerations

### ROS Environment Security
- Keep ROS Noetic and all dependencies updated
- Follow ROS security best practices
- Regularly check ROS Security Advisories (ROSSA)
- Use secure communication protocols when available

### Application Security
- Validate all input map files before processing
- Sanitize file paths and user inputs
- Handle errors gracefully without exposing system information
- Implement proper access controls for map modifications

### System Requirements
- Use only trusted Python packages from official sources
- Keep PyQt5, OpenGL, and other dependencies updated
- Implement proper file permissions for map storage
- Regular system and dependency updates

## Reporting a Vulnerability

If you discover a security vulnerability, please follow these steps:

1. **Do Not** create a public GitHub issue
2. Send details to [danialvishwa543@gmail.com]
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggestions for mitigation (if any)

### What to Expect
- Acknowledgment within 48 hours
- Regular updates on progress
- Credit for responsible disclosure
- Details kept confidential until fixed

### Timeline
1. Report received
2. Acknowledgment sent (within 48 hours)
3. Investigation and validation (1-2 weeks)
4. Fix development and testing (1-4 weeks)
5. Release and disclosure

## Best Practices for Users

1. **Installation Security**
   - Use virtual environments
   - Verify package integrity
   - Follow principle of least privilege

2. **Runtime Security**
   - Run with appropriate permissions
   - Validate input map files
   - Back up important maps

3. **Network Security**
   - Use secure network configurations
   - Implement proper firewalls
   - Follow ROS network security guidelines

## Acknowledgments

We appreciate the security research community and will acknowledge all security contributors in our releases.

## Updates

This security policy will be updated as needed. Users should check back regularly for any changes.
